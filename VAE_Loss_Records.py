"""
Shared loss-record parsing and Pareto frontier construction.

Before this module existed, four copies of ``read_loss_records`` and five
copies of the frontier algorithm lived in separate analysis scripts. The
copies had drifted apart in ways that changed which records reached the
frontier:

  - one stopped reading at the first malformed row while the others skipped
    it and continued;
  - two rejected rows by searching the raw line for the substrings "nan" and
    "inf" rather than checking parsed values;
  - two required kl_loss > 0 and two did not, even though log10(kl_loss) is
    taken downstream;
  - "iteration" meant the index among kept records in three copies and the
    index over all data lines in the fourth.

Everything now routes through here. The resolved behavior is:

  - a malformed, incomplete, or non-finite row is skipped, not fatal;
  - rows are validated by parsed value, never by substring;
  - kl_loss must be strictly positive by default;
  - "iteration" is the zero-based index among kept records, with the source
    line number retained separately as "line_number".
"""

import math
from pathlib import Path


LOSS_FILE_NAME = "losses_file.txt"


def resolve_loss_file(path):
    """
    Accept either an experiment directory or the loss file itself.

    Analysis scripts historically passed an experiment directory while
    ParetoRunSummary passed the file, so both are supported here.
    """
    path = Path(path)
    if path.is_dir():
        return path / "stats" / LOSS_FILE_NAME
    return path


def read_loss_records(path, require_positive_kl=True, warn=True):
    """
    Read chronological loss records from a losses_file.txt.

    Returns a list of dicts with keys: iteration, line_number, epoch, step,
    recon_loss, kl_loss, and kl_weight (None when the column is absent or
    unparseable).
    """
    loss_file = resolve_loss_file(path)

    if not loss_file.is_file():
        raise FileNotFoundError(f"Loss file does not exist: {loss_file}")

    records = []
    skipped = 0

    with loss_file.open("r", encoding="utf-8") as fh:
        next(fh, None)  # header

        for line_number, line in enumerate(fh, start=2):
            if not line.strip():
                continue

            fields = [field.strip() for field in line.split("--")]
            if len(fields) < 4:
                skipped += 1
                continue

            try:
                epoch = int(fields[0])
                step = int(fields[1])
                recon_loss = float(fields[2])
                kl_loss = float(fields[3])
            except ValueError:
                skipped += 1
                continue

            # The kl_weight column is followed by whitespace-separated
            # diagnostics on the same field, so only the leading token is
            # numeric. Its absence is not an error.
            kl_weight = None
            if len(fields) >= 5:
                try:
                    kl_weight = float(fields[4].split()[0])
                except (ValueError, IndexError):
                    kl_weight = None

            if not (math.isfinite(recon_loss) and math.isfinite(kl_loss)):
                skipped += 1
                continue

            if kl_weight is not None and not math.isfinite(kl_weight):
                kl_weight = None

            if require_positive_kl and kl_loss <= 0.0:
                skipped += 1
                continue

            records.append({
                "iteration": len(records),
                "line_number": line_number,
                "epoch": epoch,
                "step": step,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "kl_weight": kl_weight,
            })

    if warn and skipped:
        print(
            f"Warning: skipped {skipped} unusable loss "
            f"record(s) in {loss_file}"
        )

    if not records:
        raise ValueError(f"No valid loss records found in {loss_file}")

    return records


def pareto_records(records):
    """
    Return the records on the exact two-objective minimization frontier.

    Sort by reconstruction loss, keep the lowest KL loss among records
    sharing a reconstruction loss, then retain only records that strictly
    improve on the best KL loss seen so far. This matches
    ``VAE_ParetoFront.ParetoFront`` in exact mode.

    Ties are broken by "iteration" when present so the selection is
    deterministic and prefers the earlier record.
    """
    ordered = sorted(
        records,
        key=lambda record: (
            record["recon_loss"],
            record["kl_loss"],
            record.get("iteration", 0),
        ),
    )

    collapsed = []
    for record in ordered:
        if collapsed and record["recon_loss"] == collapsed[-1]["recon_loss"]:
            if record["kl_loss"] < collapsed[-1]["kl_loss"]:
                collapsed[-1] = record
            continue
        collapsed.append(record)

    frontier = []
    best_kl = math.inf

    for record in collapsed:
        if record["kl_loss"] < best_kl:
            frontier.append(record)
            best_kl = record["kl_loss"]

    return frontier
