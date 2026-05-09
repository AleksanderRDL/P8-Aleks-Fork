import csv
import argparse


def filter_csv(input_file, output_file="output2.csv", id_column="MMSI", ids_to_keep={"210071000"}):
    """Filter a CSV file based on a list of IDs."""
    with open(input_file, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if id_column not in fieldnames:
            print(f"Error: Column '{id_column}' not found in CSV. Available columns: {fieldnames}")
            return

        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            kept = 0
            total = 0
            for row in reader:
                total += 1
                if row[id_column] in ids_to_keep:
                    writer.writerow(row)
                    kept += 1

    print(f"Filtered {kept}/{total} rows. Output written to '{output_file}'.")


def load_ids_from_file(filepath):
    """Load IDs from a text file (one ID per line)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a raw AIS CSV by MMSI.")
    parser.add_argument("input_file", help="Path to raw AIS CSV, usually under AISDATA/raw/")
    parser.add_argument("--output-file", default="output2.csv", help="Output CSV path")
    parser.add_argument("--id-column", default="MMSI", help="Identifier column to filter on")
    parser.add_argument("--ids", nargs="+", default=["210071000"], help="MMSI values to keep")
    args = parser.parse_args()
    filter_csv(args.input_file, args.output_file, args.id_column, set(args.ids))
