# Using the data on DTU HPC

We've asked HPC support for a shared folder, and it doesn't exist yet. Until it does, each of us works in our own folder and fetches the data with `scripts/get_data.py`. When the folder is ready, replace `<SHARED_FOLDER>` below with its path.

## Connect

Log in with your DTU username. Off campus you may need the DTU VPN or two-factor authentication.

```bash
ssh <your-username>@login1.hpc.dtu.dk
```

Copy files through the transfer host, which is `transfer.gbar.dtu.dk`.

## Check your access

```bash
id -Gn
```

```bash
ls -ld <SHARED_FOLDER>
```

The first command should list the project group, and the second should show a folder you can read. If either fails, ask HPC support at support@hpc.dtu.dk to add you to the group.

## Work in the shared folder

Run this in every shell, or add it to `~/.bashrc`. Without it your teammates can delete your files but can't overwrite them.

```bash
umask 007
```

Point the download script at the folder. Anything already there is skipped, so this won't download it twice.

```bash
python3 scripts/get_data.py --data-dir <SHARED_FOLDER>/data --status
```

If you use a coding agent, tell it the folder path and to follow [AGENTS.md](../AGENTS.md).

## Run jobs

Login nodes are for editing and submitting only. Submit real work as a batch job to a GPU queue such as `gpua100` or `gpuv100`, and check it with `bjobs`. A job can run for 24 hours at most.

These commands only work in a login shell. If you run them through `ssh` from your laptop, wrap them in `bash -lc "..."`.

## Rules for the data

- Keep screenplays, reviews and article text in the shared folder or a private one, and never in a folder that people outside the group can read.
- Don't commit them to this repo.
- Set the folder's group permissions so new files stay in the project group.
