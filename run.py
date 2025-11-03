import argparse

from config import dataset_paths, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    
    args = parser.parse_args()

    dataset_path = dataset_paths[args.dataset]
    ModelClass = models[args.model]  # Get class
    model = ModelClass()             # Instantiate only requested model

    output_csv = model.inference_tsv(dataset_path, debug_mode=args.debug)
    model.eval_results(output_csv, args.dataset, debug_mode=args.debug)

if __name__ == "__main__":
    main()