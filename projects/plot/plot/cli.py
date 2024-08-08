import os
import yaml
from plot.plot import LivePlotter

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_file = 'config.yaml'
    config = load_config(config_file)
    
    log_dir = config.get('log_directory', '.')
    plot_dir = config.get('plot_directory', '.')
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plotter = LivePlotter(log_dir=log_dir, output_plot_pattern=os.path.join(plot_dir, "plot_{date}.png"))
    plotter.continuous_update(interval=60)

if __name__ == "__main__":
    main()

