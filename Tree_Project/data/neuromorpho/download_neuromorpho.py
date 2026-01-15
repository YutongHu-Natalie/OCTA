"""
NeuroMorpho.org Data Downloader

Downloads neuron morphology data for replicating the KDD24 Geometric Trees paper.

Paper's binary classification tasks:
- lps-glia: LPS-treated glia vs Control glia (28,687 trees)
- lps-inter: LPS-treated interneurons vs Control interneurons (8,092 trees)
- lps-pc: LPS-treated principal cells vs Control principal cells (28,224 trees)
- 5xfad-glia: 5xFAD glia vs Control glia (33,757 trees)
- 5xfad-pc: 5xFAD principal cells vs Control principal cells (28,086 trees)

API Reference: https://neuromorpho.org/apiReference.html
"""

import os
import json
import time
import requests
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from dataclasses import dataclass

# API Configuration
BASE_URL = "https://neuromorpho.org/api"
SWC_URL_TEMPLATE = "https://neuromorpho.org/dableFiles/{archive}/CNG version/{neuron_name}.CNG.swc"
SWC_SOURCE_TEMPLATE = "https://neuromorpho.org/dableFiles/{archive}/Source-Version/{neuron_name}.swc"

# Rate limiting
REQUEST_DELAY = 0.05  # seconds between API calls
MAX_RETRIES = 3

# Experiment conditions as they appear in NeuroMorpho.org
CONDITIONS = {
    "lps": "lipopolysaccharide injection",
    "5xfad": "5xFAD Alzheimer model (APP overexpression)",
    "control": "Control"
}

# Cell types as they appear in NeuroMorpho.org
CELL_TYPES = {
    "glia": ["Glia", "microglia", "astrocyte"],  # Include subtypes
    "interneuron": ["interneuron"],
    "principal_cell": ["principal cell", "pyramidal"]  # Include pyramidal as subtype
}


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    condition: str  # Key from CONDITIONS
    control: str    # Key from CONDITIONS (usually "control")
    cell_types: List[str]  # Keys from CELL_TYPES
    label_condition: int  # Label for condition samples
    label_control: int    # Label for control samples


# Paper's dataset configurations
PAPER_DATASETS = [
    DatasetConfig("lps-glia", "lps", "control", ["glia"], label_condition=1, label_control=0),
    DatasetConfig("lps-inter", "lps", "control", ["interneuron"], label_condition=1, label_control=0),
    DatasetConfig("lps-pc", "lps", "control", ["principal_cell"], label_condition=1, label_control=0),
    DatasetConfig("5xfad-glia", "5xfad", "control", ["glia"], label_condition=1, label_control=0),
    DatasetConfig("5xfad-pc", "5xfad", "control", ["principal_cell"], label_condition=1, label_control=0),
]


class NeuroMorphoDownloader:
    def __init__(self, output_dir: str, use_cng: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cng = use_cng
        self.session = requests.Session()

    def _make_request(self, url: str, params: dict = None) -> Optional[dict]:
        """Make API request with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                time.sleep(REQUEST_DELAY)
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                else:
                    print(f"  Request failed: {e}")
        return None

    def search_neurons(self, experiment_condition: str, cell_types: List[str],
                       species: str = "mouse", max_results: int = None) -> List[Dict]:
        """
        Search for neurons matching criteria.

        Args:
            experiment_condition: Condition string (e.g., "lipopolysaccharide injection")
            cell_types: List of cell type strings
            species: Species to filter by
            max_results: Maximum number of results

        Returns:
            List of neuron records
        """
        all_neurons = []

        for cell_type in cell_types:
            print(f"  Searching for {cell_type}...")
            page = 0
            page_size = 500

            while True:
                # Build query URL
                query = f"species:{species}"
                filters = [
                    f"experiment_condition:{experiment_condition}",
                    f"cell_type:{cell_type}"
                ]
                filter_str = "&".join([f"fq={urllib.parse.quote(f)}" for f in filters])
                url = f"{BASE_URL}/neuron/select?q={urllib.parse.quote(query)}&{filter_str}&page={page}&size={page_size}"

                data = self._make_request(url)
                if not data or "_embedded" not in data:
                    break

                neurons = data["_embedded"].get("neuronResources", [])
                all_neurons.extend(neurons)

                # Check if more pages
                page_info = data.get("page", {})
                total = page_info.get("totalElements", 0)
                if page == 0:
                    print(f"    Found {total} {cell_type} neurons")

                if page >= page_info.get("totalPages", 1) - 1:
                    break
                if max_results and len(all_neurons) >= max_results:
                    break

                page += 1

        if max_results:
            all_neurons = all_neurons[:max_results]

        return all_neurons

    def download_swc(self, neuron: Dict, output_path: Path) -> bool:
        """Download SWC file for a neuron."""
        archive = neuron.get("archive", "").lower()
        neuron_name = neuron.get("neuron_name", "")

        if not archive or not neuron_name:
            return False

        # Try CNG version first, then source version
        urls_to_try = []
        if self.use_cng:
            urls_to_try.append(
                SWC_URL_TEMPLATE.format(archive=archive, neuron_name=neuron_name)
            )
        urls_to_try.append(
            SWC_SOURCE_TEMPLATE.format(archive=archive, neuron_name=neuron_name)
        )

        for url in urls_to_try:
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200 and not response.text.startswith("<!DOCTYPE"):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(response.text)
                    time.sleep(REQUEST_DELAY)
                    return True
            except Exception:
                continue

        return False

    def download_dataset(self, config: DatasetConfig, max_per_class: int = None) -> Dict:
        """
        Download a complete dataset for binary classification.

        Args:
            config: Dataset configuration
            max_per_class: Maximum neurons per class (for testing)

        Returns:
            Statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Downloading dataset: {config.name}")
        print(f"Condition: {config.condition} vs {config.control}")
        print(f"Cell types: {config.cell_types}")
        print(f"{'='*60}")

        # Create dataset directory
        dataset_dir = self.output_dir / config.name
        swc_dir = dataset_dir / "swc"
        swc_dir.mkdir(parents=True, exist_ok=True)

        # Get cell type strings
        cell_type_strs = []
        for ct_key in config.cell_types:
            cell_type_strs.extend(CELL_TYPES.get(ct_key, [ct_key]))

        # Download condition samples (label=1)
        print(f"\nSearching for {config.condition} samples...")
        condition_neurons = self.search_neurons(
            experiment_condition=CONDITIONS[config.condition],
            cell_types=cell_type_strs,
            max_results=max_per_class
        )
        for n in condition_neurons:
            n["label"] = config.label_condition

        # Download control samples (label=0)
        print(f"\nSearching for {config.control} samples...")
        control_neurons = self.search_neurons(
            experiment_condition=CONDITIONS[config.control],
            cell_types=cell_type_strs,
            max_results=max_per_class
        )
        for n in control_neurons:
            n["label"] = config.label_control

        # Combine
        all_neurons = condition_neurons + control_neurons
        print(f"\nTotal neurons: {len(all_neurons)}")
        print(f"  {config.condition}: {len(condition_neurons)}")
        print(f"  {config.control}: {len(control_neurons)}")

        # Save metadata
        metadata = {
            "dataset": config.name,
            "condition": config.condition,
            "control": config.control,
            "cell_types": config.cell_types,
            "condition_count": len(condition_neurons),
            "control_count": len(control_neurons),
            "total_count": len(all_neurons),
            "neurons": all_neurons
        }
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

        # Download SWC files
        print("\nDownloading SWC files...")
        downloaded = 0
        failed = 0
        labels = []

        for i, neuron in enumerate(all_neurons):
            neuron_name = neuron.get("neuron_name", f"unknown_{i}")
            swc_path = swc_dir / f"{neuron_name}.swc"

            if swc_path.exists():
                downloaded += 1
                labels.append({"name": neuron_name, "label": neuron["label"]})
                continue

            if self.download_swc(neuron, swc_path):
                downloaded += 1
                labels.append({"name": neuron_name, "label": neuron["label"]})
            else:
                failed += 1

            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{len(all_neurons)} "
                      f"(downloaded: {downloaded}, failed: {failed})")

        # Save labels
        labels_path = dataset_dir / "labels.json"
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)

        print(f"\nDataset {config.name} complete:")
        print(f"  Total found: {len(all_neurons)}")
        print(f"  Downloaded: {downloaded}")
        print(f"  Failed: {failed}")

        stats = {
            "name": config.name,
            "condition_found": len(condition_neurons),
            "control_found": len(control_neurons),
            "total_found": len(all_neurons),
            "downloaded": downloaded,
            "failed": failed
        }

        stats_path = dataset_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        return stats


def explore_database():
    """Explore available fields and values in the database."""
    print("="*60)
    print("NeuroMorpho.org Database Exploration")
    print("="*60)

    session = requests.Session()

    # Check experiment conditions
    print("\nExperiment Conditions (relevant to paper):")
    url = f"{BASE_URL}/neuron/fields/experiment_condition?page=0&size=200"
    data = session.get(url).json()
    relevant = ["control", "lps", "5xfad", "alzheimer"]
    for field in data.get("fields", []):
        if any(r in field.lower() for r in relevant):
            print(f"  - {field}")

    # Check cell types
    print("\nCell Types (relevant to paper):")
    url = f"{BASE_URL}/neuron/fields/cell_type?page=0&size=200"
    data = session.get(url).json()
    relevant = ["glia", "intern", "principal", "pyramid", "astro", "micro"]
    for field in data.get("fields", []):
        if any(r in field.lower() for r in relevant):
            print(f"  - {field}")


def count_datasets():
    """Count neurons available for each paper dataset."""
    print("="*60)
    print("Dataset Counts (matching paper's binary classification)")
    print("="*60)

    session = requests.Session()

    for config in PAPER_DATASETS:
        print(f"\n{config.name}:")

        # Get cell type strings
        cell_type_strs = []
        for ct_key in config.cell_types:
            cell_type_strs.extend(CELL_TYPES.get(ct_key, [ct_key]))

        condition_count = 0
        control_count = 0

        for cell_type in cell_type_strs:
            # Count condition
            query = f"species:mouse"
            filters = [
                f"experiment_condition:{CONDITIONS[config.condition]}",
                f"cell_type:{cell_type}"
            ]
            filter_str = "&".join([f"fq={urllib.parse.quote(f)}" for f in filters])
            url = f"{BASE_URL}/neuron/select?q={urllib.parse.quote(query)}&{filter_str}&size=1"
            data = session.get(url).json()
            condition_count += data.get("page", {}).get("totalElements", 0)

            # Count control
            filters = [
                f"experiment_condition:{CONDITIONS[config.control]}",
                f"cell_type:{cell_type}"
            ]
            filter_str = "&".join([f"fq={urllib.parse.quote(f)}" for f in filters])
            url = f"{BASE_URL}/neuron/select?q={urllib.parse.quote(query)}&{filter_str}&size=1"
            data = session.get(url).json()
            control_count += data.get("page", {}).get("totalElements", 0)

        print(f"  {config.condition}: {condition_count}")
        print(f"  {config.control}: {control_count}")
        print(f"  Total: {condition_count + control_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Download neuron data from NeuroMorpho.org for Geometric Trees replication"
    )
    parser.add_argument(
        "--output", "-o",
        default="./downloaded_data",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Explore database fields and values"
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Count available neurons for each dataset"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download paper datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["lps-glia", "lps-inter", "lps-pc", "5xfad-glia", "5xfad-pc", "all"],
        default="all",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum neurons per class (for testing)"
    )
    parser.add_argument(
        "--use-source",
        action="store_true",
        help="Use source version instead of CNG standardized"
    )

    args = parser.parse_args()

    if args.explore:
        explore_database()

    if args.count:
        count_datasets()

    if args.download:
        downloader = NeuroMorphoDownloader(
            output_dir=args.output,
            use_cng=not args.use_source
        )

        datasets_to_download = PAPER_DATASETS
        if args.dataset != "all":
            datasets_to_download = [d for d in PAPER_DATASETS if d.name == args.dataset]

        all_stats = []
        for config in datasets_to_download:
            stats = downloader.download_dataset(config, args.max_per_class)
            all_stats.append(stats)

        # Summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        for stats in all_stats:
            print(f"{stats['name']}: {stats['downloaded']}/{stats['total_found']} downloaded")

    if not args.explore and not args.count and not args.download:
        print("NeuroMorpho.org Data Downloader")
        print("="*40)
        print("Options:")
        print("  --explore     : See available fields/values")
        print("  --count       : Count neurons for each dataset")
        print("  --download    : Download the data")
        print("  --help        : All options")
        print()
        print("Example usage:")
        print("  python download_neuromorpho.py --count")
        print("  python download_neuromorpho.py --download --dataset lps-glia --max-per-class 100")
        print("  python download_neuromorpho.py --download --dataset all")


if __name__ == "__main__":
    main()
