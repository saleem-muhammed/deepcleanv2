import argparse
import yaml
import os
from clean.data import DeepCleanInferenceDataset
from clean.infer import OnlineInference
from clean.model import InferenceModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run online inference process.")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the configuration YAML file.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Load configuration from YAML
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    

    # Initialize InferenceModel
    model = InferenceModel(config['train_dir'], config['sample_rate'], config['device'])
    
    # Initialize DeepCleanInferenceDataset
    inference_dataset = DeepCleanInferenceDataset(
        hoft_dir=config['hoft_dir'],
        witness_dir=config['witness_dir'],
        model=model,
        device=config['device']
    )
    
    # Initialize OnlineInference
    online_inference = OnlineInference(
        dataset=inference_dataset,
        model=model,
        outdir=config['outdir'],
        device=config['device']
    )
    
    # Run the online inference process for a number of iterations (e.g., 100)
    # for k in range(300):
    while True:
        online_inference.predict_and_write()
        online_inference.dataset.update()
        #print(f"iteration {k}")

if __name__ == "__main__":
    main()
