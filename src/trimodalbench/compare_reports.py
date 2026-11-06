"""Summarize RGB–Flow evaluation reports as a CSV and PNG."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def comparison_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for frame in report["frames"]:
            regions = [
                ("all_valid", frame["all_valid"], frame["rgb_flow"]["all_valid"],
                 frame["depth_reprojection"]["all_valid"]["primary"]),
            ]
            if frame["static"] is not None:
                regions.append(("static", frame["static"], frame["rgb_flow"]["static"],
                                frame["depth_reprojection"]["static"]["primary"]))
            if "moving_roi" in frame:
                moving = frame["moving_roi"]
                regions.append(("moving_roi", moving["flow_geometry"], moving["rgb_flow"], None))
            for region, geometry, rgb, depth in regions:
                rows.append({
                    "sequence": report["sequence"],
                    "source_frame": frame["source_frame"],
                    "target_frame": frame["target_frame"],
                    "region": region,
                    "rgb_flow_mean_intensity": rgb["mean_intensity"],
                    "identity_mean_intensity": rgb["identity_mean_intensity"],
                    "alignment_gain": rgb["alignment_gain"],
                    "rgb_coverage": rgb["coverage"],
                    "flow_geometry_median_px": geometry["median_px"],
                    "depth_reprojection_median_m": None if depth is None else depth["absolute"]["median_m"],
                })
    return rows


def save_summary(rows: list[dict], path: Path) -> None:
    data = [row for row in rows if row["region"] == "all_valid"]
    image = Image.new("RGB", (760, 155 + 57 * len(data)), "#f7fafc")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=24)
    body = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=13)
    ink, flow, identity = "#172b40", "#087f8c", "#637995"
    draw.text((30, 22), "Sintel photometric alignment", font=title, fill=ink)
    draw.text((30, 60), "Mean Charbonnier residual on matched support (lower is better)", font=body, fill=ink)
    draw.text((194, 96), "Forward flow", font=body, fill=flow)
    draw.text((326, 96), "Identity", font=body, fill=identity)
    draw.text((654, 96), "Gain", font=body, fill=ink)
    keys = ("rgb_flow_mean_intensity", "identity_mean_intensity")
    maximum = max((row[key] for row in data for key in keys if row[key] is not None), default=1)
    for index, row in enumerate(data):
        y = 132 + index * 57
        label = f"{row['sequence']}  {row['source_frame']} -> {row['target_frame']}"
        draw.text((30, y + 9), label, font=body, fill=ink)
        for offset, key, color in zip((0, 18), keys, (flow, identity)):
            value = row[key]
            if value is not None:
                right = 194 + round(350 * value / maximum)
                draw.rectangle((194, y + offset, right, y + offset + 12), fill=color)
                draw.text((right + 8, y + offset - 2), f"{value:.4f}", font=small, fill=ink)
        gain = row["alignment_gain"]
        draw.text((654, y + 9), f"{gain:.1%}" if gain is not None else "n/a", font=body, fill=ink)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = comparison_rows(args.reports)
    if not rows:
        parser.error("No frame results in the supplied reports")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metric_comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_summary(rows, args.output / "summary.png")


if __name__ == "__main__":
    main()
