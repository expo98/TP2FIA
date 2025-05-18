import sys

import matplotlib.pyplot as plt

def read_values(filename):
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <datafile>")
        sys.exit(1)

    filename = sys.argv[1]
    y_values = read_values(filename)
    x_values = list(range(len(y_values)))

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, marker='o')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Values from File')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
