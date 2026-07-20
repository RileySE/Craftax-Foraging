#!/usr/bin/env python3
"""Run a slurm training job to completion across wallclock limits.

Given an sbatch submission script (e.g. example_slurm_script.sh), this tool
submits it and keeps watch; whenever the job dies before training finishes
(typically at the 48 h time limit), it submits a continuation job that resumes
from the latest checkpoint, until every (array) task has trained to its final
update.

Usage:

    python utils/auto_resume.py my_job.sh

Run it from the same directory you would run `sbatch my_job.sh` from (slurm
jobs execute in the submission directory, and scripts often use relative
paths), and keep it alive for the duration of the run, e.g. inside tmux/screen
or with:

    nohup python utils/auto_resume.py my_job.sh > my_job.autoresume.out 2>&1 &

How it works
------------
* The sbatch script is submitted UNMODIFIED. Resumability is switched on by
  exporting PPO_AUTO_RESUME=1 into the job's environment, which makes
  forageworld/ppo_rnn.py resume from the newest checkpoint under its
  --output_path (or start fresh when there is none) and continue the original
  WandB run. Checkpointing must be enabled (--checkpoint_interval > 0, the
  default).
* A state directory, <script>.auto_resume/, is created next to the script.
  When a training task runs to its final update it writes a done_<task-id>
  marker there (ppo_rnn.py receives the location via PPO_AUTO_RESUME_STATE_DIR
  in the job environment); tasks with a marker are excluded from later
  submissions by narrowing the job array with `sbatch --array=<remaining>`,
  which overrides the script's own #SBATCH --array directive.
* Whenever the current job leaves the queue, tasks without a done marker are
  resubmitted. Two safety valves prevent futile resubmission: a task whose
  last leg was CANCELLED is dropped (a scancel is assumed to be deliberate),
  and a task is given up on after --max-consecutive-failures legs in a row
  that ended without finishing and either ran shorter than --min-leg-seconds
  or exited "successfully" without writing a marker (i.e. the job crashed
  immediately, or the training command never reports completion). Timeouts
  and long-running failed legs are the expected case and are always
  resubmitted, up to --max-restarts legs per task overall.
* The manager may be stopped and restarted freely: done markers persist in the
  state directory, and if the leg it submitted is still queued or running the
  restarted manager re-attaches to it instead of submitting a duplicate. To
  rerun an experiment from scratch, delete its checkpoints AND pass --fresh
  (or delete the state directory).

Tip: use %A_%a patterns in the script's #SBATCH --output/--error paths so
continuation legs (and array tasks) do not overwrite one another's logs.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from hashlib import sha256

# States sacct reports for jobs that are still going somewhere; anything else
# is terminal. REQUEUED/PENDING matter: a node failure can requeue a leg under
# the same job id, and it must then be waited on rather than resubmitted.
ACTIVE_STATES = {"PENDING", "RUNNING", "COMPLETING", "REQUEUED", "SUSPENDED", "RESIZING"}

LOG_PATH = None  # set in main(); log() mirrors everything into the state dir


def log(message):
    line = "[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a") as log_file:
            log_file.write(line + "\n")


def parse_sbatch_option(script_text, name):
    """Return the value of an #SBATCH option from a submission script, or None.
    Handles both `#SBATCH --array=0-4` and `#SBATCH --array 0-4`; the last
    occurrence wins, matching sbatch.
    """
    pattern = re.compile(
        r"^\s*#SBATCH\s+--" + re.escape(name) + r"(?:=|\s+)(\S+)", re.MULTILINE
    )
    matches = pattern.findall(script_text)
    return matches[-1] if matches else None


def expand_array_spec(spec):
    """Expand an sbatch array spec into task ids: '0-4%2' -> ([0,1,2,3,4], '%2').
    Supports comma lists, a-b ranges, and a-b:step ranges, with an optional
    %limit throttle suffix (preserved so resubmissions keep the throttle).
    """
    throttle = ""
    if "%" in spec:
        spec, limit = spec.split("%", 1)
        throttle = "%" + limit
    ids = set()
    for item in spec.split(","):
        step = 1
        if ":" in item:
            item, step_text = item.split(":", 1)
            step = int(step_text)
        if "-" in item:
            low, high = item.split("-", 1)
            ids.update(range(int(low), int(high) + 1, step))
        else:
            ids.add(int(item))
    return sorted(ids), throttle


def parse_elapsed(text):
    """sacct Elapsed ([D-]HH:MM:SS) -> seconds, or None if unparsable."""
    match = re.match(r"^(?:(\d+)-)?(\d+):(\d{2}):(\d{2})$", text.strip())
    if not match:
        return None
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def run_command(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def sacct_task_states(job_id, is_array):
    """Map task id (str) -> (state, elapsed_seconds) via sacct, or None when
    sacct is unavailable/empty (accounting disabled or not yet flushed)."""
    result = run_command(
        ["sacct", "-j", job_id, "-X", "--noheader", "--parsable2",
         "--format=JobID,State,Elapsed"]
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    states = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        row_id, state, elapsed = parts[0], parts[1], parts[2]
        state = state.split()[0]  # "CANCELLED by 1234" -> "CANCELLED"
        elapsed = parse_elapsed(elapsed)
        if not is_array:
            if row_id == job_id:
                states["noarray"] = (state, elapsed)
            continue
        match = re.match(r"^\d+_(\d+)$", row_id)
        if match:
            states[match.group(1)] = (state, elapsed)
            continue
        # Tasks that never started are aggregated as e.g. "1234_[2-4%1]".
        match = re.match(r"^\d+_\[(.+)\]$", row_id)
        if match:
            task_ids, _ = expand_array_spec(match.group(1))
            for task in task_ids:
                states[str(task)] = (state, elapsed)
    return states or None


def leg_is_active(job_id, is_array):
    """True while the leg is queued/running/requeued, False once every task is
    terminal, None when slurm could not be queried (treat as still active)."""
    result = run_command(["squeue", "-h", "-j", job_id, "-o", "%i"])
    if result.returncode == 0:
        if result.stdout.strip():
            return True
    elif "invalid job id" not in (result.stderr or "").lower():
        return None  # transient squeue failure; not evidence the job is done
    # Not in the queue (or already aged out of it): confirm via accounting
    # that no task is PENDING/REQUEUED/etc. before declaring the leg over.
    states = sacct_task_states(job_id, is_array)
    if states is None:
        return False  # no accounting available; squeue's word is final
    return any(state in ACTIVE_STATES for state, _ in states.values())


def wait_for_leg(job_id, is_array, poll_interval):
    consecutive_errors = 0
    while True:
        time.sleep(poll_interval)
        active = leg_is_active(job_id, is_array)
        if active is False:
            return
        if active is None:
            consecutive_errors += 1
            if consecutive_errors % 5 == 1:
                log("warning: cannot query slurm for job {} (attempt {}); "
                    "will keep retrying".format(job_id, consecutive_errors))
        else:
            consecutive_errors = 0


def submit_leg(script_path, state_dir, task_ids, throttle, is_array):
    """Submit one leg, returning the job id. Raises RuntimeError on failure."""
    cmd = [
        "sbatch", "--parsable",
        "--export=ALL,PPO_AUTO_RESUME=1,PPO_AUTO_RESUME_STATE_DIR={}".format(state_dir),
    ]
    if is_array:
        cmd.append("--array={}{}".format(",".join(task_ids), throttle))
    cmd.append(script_path)
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            "sbatch failed: {}".format(result.stderr.strip() or result.stdout.strip())
        )
    return result.stdout.strip().splitlines()[-1].split(";")[0]


def main():
    global LOG_PATH

    parser = argparse.ArgumentParser(
        description="Submit a slurm training job and automatically submit "
                    "continuation jobs (resuming from checkpoints) until every "
                    "array task has trained to completion. See the module "
                    "docstring for details.",
    )
    parser.add_argument("script", help="sbatch submission script to run to completion")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="seconds between slurm queue checks (default: %(default)s)")
    parser.add_argument("--max-restarts", type=int, default=60,
                        help="hard cap on legs submitted per task (default: %(default)s)")
    parser.add_argument("--max-consecutive-failures", type=int, default=3,
                        help="give up on a task after this many consecutive legs that "
                             "ended quickly without finishing (default: %(default)s)")
    parser.add_argument("--min-leg-seconds", type=int, default=600,
                        help="a leg shorter than this that did not finish counts as a "
                             "failure; longer legs are assumed to have made checkpoint "
                             "progress (default: %(default)s)")
    parser.add_argument("--fresh", action="store_true",
                        help="clear this script's auto-resume state (done markers, "
                             "tracked job) before starting; use together with deleting "
                             "the run's checkpoints to restart an experiment from scratch")
    args = parser.parse_args()

    script_path = os.path.abspath(args.script)
    if not os.path.isfile(script_path):
        parser.error("no such script: {}".format(script_path))
    with open(script_path) as script_file:
        script_text = script_file.read()

    state_dir = script_path + ".auto_resume"
    os.makedirs(state_dir, exist_ok=True)
    LOG_PATH = os.path.join(state_dir, "manager.log")
    current_job_path = os.path.join(state_dir, "current_job")
    script_hash_path = os.path.join(state_dir, "script.sha256")

    if args.fresh:
        removed = []
        for name in sorted(os.listdir(state_dir)):
            if name.startswith("done_") or name in ("current_job", "script.sha256"):
                os.remove(os.path.join(state_dir, name))
                removed.append(name)
        log("--fresh: cleared state {} in {}".format(removed or "(none)", state_dir))

    # A changed script with leftover done markers can silently skip tasks the
    # user meant to rerun (e.g. after changing --output_path). Warn loudly.
    script_hash = sha256(script_text.encode()).hexdigest()
    if os.path.exists(script_hash_path):
        with open(script_hash_path) as hash_file:
            if hash_file.read().strip() != script_hash:
                log("WARNING: {} changed since this state directory was created. "
                    "Existing done markers will still be honored; if the change "
                    "altered output paths or made old results invalid, rerun with "
                    "--fresh (and delete stale checkpoints).".format(args.script))
    with open(script_hash_path, "w") as hash_file:
        hash_file.write(script_hash + "\n")

    array_spec = parse_sbatch_option(script_text, "array")
    is_array = array_spec is not None
    if is_array:
        array_ids, throttle = expand_array_spec(array_spec)
        task_ids = [str(task) for task in array_ids]
    else:
        task_ids, throttle = ["noarray"], ""
    job_name = parse_sbatch_option(script_text, "job-name") or os.path.basename(script_path)

    log("managing '{}' ({}); state dir: {}".format(
        job_name,
        "array tasks {}".format(",".join(task_ids)) if is_array else "single job",
        state_dir))
    log("this process must outlive the jobs it manages - run it under "
        "tmux/screen/nohup; it is safe to stop and rerun (it re-attaches)")

    finished = {task for task in task_ids
                if os.path.exists(os.path.join(state_dir, "done_" + task))}
    if finished:
        log("already finished in a previous session: {}".format(",".join(sorted(finished))))
    abandoned = {}  # task id -> reason
    legs_run = defaultdict(int)
    consecutive_failures = defaultdict(int)

    # Re-attach to a leg submitted by a previous manager instance, if any.
    job_id = None
    if os.path.exists(current_job_path):
        with open(current_job_path) as job_file:
            previous_job = json.load(job_file).get("job_id")
        if previous_job and leg_is_active(previous_job, is_array):
            job_id = previous_job
            log("re-attaching to job {} (still in the queue)".format(job_id))

    while True:
        remaining = [task for task in task_ids
                     if task not in finished and task not in abandoned]
        if not remaining:
            break

        if job_id is None:
            attempt = 0
            while True:
                try:
                    job_id = submit_leg(script_path, state_dir, remaining, throttle, is_array)
                    break
                except RuntimeError as error:
                    # A first-ever submission failing is almost certainly a
                    # configuration problem: surface it immediately. Mid-run,
                    # ride out scheduler hiccups with capped backoff.
                    if not any(legs_run.values()):
                        log("ERROR: {}".format(error))
                        sys.exit(1)
                    attempt += 1
                    if attempt >= 10:
                        log("ERROR: giving up after {} failed submissions: {}".format(attempt, error))
                        sys.exit(1)
                    delay = min(args.poll_interval * attempt, 1800)
                    log("warning: {} (retrying in {} s)".format(error, delay))
                    time.sleep(delay)
            with open(current_job_path, "w") as job_file:
                json.dump({"job_id": job_id, "tasks": remaining}, job_file)
            for task in remaining:
                legs_run[task] += 1
            log("submitted job {} for task(s) {} (leg {} for task {})".format(
                job_id, ",".join(remaining), legs_run[remaining[0]], remaining[0]))

        wait_for_leg(job_id, is_array, args.poll_interval)
        log("job {} left the queue; checking results".format(job_id))
        # Grace period for done markers written on a compute node to appear on
        # this (shared) filesystem.
        time.sleep(15)
        states = sacct_task_states(job_id, is_array)
        if states is None:
            log("warning: no accounting data for job {}; relying on done markers only".format(job_id))
            states = {}

        for task in remaining:
            if os.path.exists(os.path.join(state_dir, "done_" + task)):
                finished.add(task)
                log("task {} finished training".format(task))
                continue
            state, elapsed = states.get(task, ("UNKNOWN", None))
            if state == "CANCELLED":
                abandoned[task] = "leg was cancelled (scancel is assumed deliberate)"
                log("task {}: {} - not resubmitting".format(task, abandoned[task]))
                continue
            if state == "COMPLETED":
                # The batch script exited zero yet training never reported
                # completion: either a later line in the script masked a crash,
                # or the training command does not support auto-resume markers.
                # Resubmitting can never end this loop, so treat it as failure.
                consecutive_failures[task] += 1
                log("warning: task {} exited successfully without a done marker "
                    "({} in a row) - does the training command support "
                    "auto-resume?".format(task, consecutive_failures[task]))
            elif elapsed is not None and elapsed < args.min_leg_seconds:
                consecutive_failures[task] += 1
                log("task {}: leg ended {} after {} s ({} quick failures in a row)".format(
                    task, state, elapsed, consecutive_failures[task]))
            else:
                # TIMEOUT is the expected way a leg ends; any long-running leg
                # presumably wrote checkpoints, so the crash-loop counter resets.
                consecutive_failures[task] = 0
                log("task {}: leg ended {} - will resume from checkpoint".format(task, state))
            if consecutive_failures[task] >= args.max_consecutive_failures:
                abandoned[task] = "{} consecutive legs ended without progress".format(
                    consecutive_failures[task])
                log("task {}: {} - giving up (fix the problem, then rerun this "
                    "command to retry)".format(task, abandoned[task]))
            elif legs_run[task] >= args.max_restarts:
                abandoned[task] = "reached --max-restarts={}".format(args.max_restarts)
                log("task {}: {} - giving up".format(task, abandoned[task]))
        job_id = None

    if os.path.exists(current_job_path):
        os.remove(current_job_path)
    log("all tasks accounted for: {} finished{}".format(
        len(finished),
        ", {} abandoned ({})".format(
            len(abandoned),
            "; ".join("task {}: {}".format(task, reason)
                      for task, reason in sorted(abandoned.items())),
        ) if abandoned else ""))
    sys.exit(1 if abandoned else 0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("manager interrupted - any submitted job is still in the queue and "
            "keeps running; rerun the same command to re-attach and continue")
        sys.exit(130)
