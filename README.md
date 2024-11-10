# RA_Assignment2

## Project Description
This project investigates load balancing and allocation strategies in a bin-packing scenario through simulations. The project consists of three Python scripts: `lab2.py`, `lab2_1.py`, and `lab2_2.py`, each implementing specific experimental setups and methods to evaluate various ball allocation strategies across bins. These experiments analyze how different strategies affect the gap, defined as the difference between the maximum and minimum load in the bins.


## Installation
To run the scripts, make sure Python 3 is installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

It’s recommended to use a virtual environment to manage dependencies:

```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
pip3 install numpy matplotlib tqdm
```

## Usage
To run each script, use the following commands in your terminal:

### lab2.py
```bash
python lab2.py
```

### lab2_1.py
```bash
python lab2_1.py
```

### lab2_2.py
```bash
python lab2_2.py
```

## File Descriptions
- **lab2.py**: This experiment uses partial information to allocate balls into bins. Depending on the value of k, different levels of information about bin loads are used in allocation decisions.
- **lab2_1.py**: This baseline experiment assumes all balls arrive simultaneously and compares several strategies to allocate the balls in a way that minimizes the load gap.
- **lab2_2.py**: This experiment explores the effect of batch sizes on allocation strategies. Balls arrive in batches, and strategies are adapted to make allocation decisions based on batched arrival.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.