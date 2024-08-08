import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

class LivePlotter:
    def __init__(self, log_dir, log_file_pattern="log_*.log", output_plot_pattern="plot_{date}.png"):
        self.log_dir = log_dir
        self.log_file_pattern = log_file_pattern
        self.output_plot_pattern = output_plot_pattern
        self.data = pd.DataFrame(columns=["output_t0", "timestamp", "mean_asd_ratio", "max_asd_ratio"])
        self.current_date = None

    def read_log_files(self):
        log_files = [os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) if f.startswith("log_")]
        all_data = []
        for log_file in log_files:
            with open(log_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.split(", ")
                    output_t0 = float(parts[0])
                    timestamp = datetime.strptime(parts[1].split(" - ")[0], '%Y-%m-%d %H:%M:%S')
                    mean_asd_ratio = float(parts[1].split(" - ")[1].split(": ")[1].split(",")[0])
                    max_asd_ratio = float(parts[1].split(" - ")[1].split(": ")[2])
                    all_data.append([output_t0, timestamp, mean_asd_ratio, max_asd_ratio])
        
        self.data = pd.DataFrame(all_data, columns=["output_t0", "timestamp", "mean_asd_ratio", "max_asd_ratio"])

    def plot_data(self):
        if self.data.empty:
            print("No data available for plotting.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.data["output_t0"], self.data["mean_asd_ratio"], label='Mean ASDR')
        plt.plot(self.data["output_t0"], self.data["max_asd_ratio"], label='Max ASDR')
        plt.xlabel('Output T0')
        plt.ylabel('ASDR Ratio')
        plt.legend()
        plt.title('ASDR Ratios Over Time')

        # Generate the plot filename based on the current date
        current_date = datetime.utcnow().strftime('%Y-%m-%d')
        output_plot = os.path.join(self.log_dir, self.output_plot_pattern.format(date=current_date))
        plt.savefig(output_plot)
        plt.close()
        print(f"Plot saved as: {output_plot}")

    def update_plot(self):
        self.read_log_files()
        self.plot_data()

    def continuous_update(self, interval=60):
        while True:
            self.update_plot()
            time.sleep(interval)

if __name__ == "__main__":
    log_dir = "/path/to/log/directory"  # Update this to the directory where your log files are stored
    plotter = LivePlotter(log_dir=log_dir)
    plotter.continuous_update(interval=60)


