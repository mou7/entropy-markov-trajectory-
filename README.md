# entropy-markov-trajectory-

To quantify the randomness of Markov trajectories with fixed initial and final states, we need to compute the entropy associated with the distribution of trajectories. The code proposed here allows for computing the matrix of trajectory entropies associated with an irreducible finite state Markov chain.

Definitions and computations can be found

  [1] The entropy of Markov trajectories 
  http://www-isl.stanford.edu/~cover/papers/paper101.pdf

The example of the MC presented in this script is from the paper

  [2] The entropy of conditional Markov trajectories 
  http://arxiv.org/abs/1212.2831

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the code with the example of Markov chain given in [2]:

```python
python trajectory_entropy.py
```

## Functions

The module provides three main functions:

- `local_entropy(P)`: Computes the local entropy at each state of the Markov chain
- `stationary_distribution(P)`: Computes the stationary distribution of the Markov chain
- `trajectory_entropy(P)`: Computes the matrix of trajectory entropies

All functions include comprehensive input validation and error handling.

## Testing

This project includes a comprehensive test suite to ensure correctness and robustness.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with short traceback format
pytest --tb=short

# Run specific test class
pytest test_trajectory_entropy.py::TestLocalEntropy -v

# Run tests and stop on first failure
pytest -x
```

### Test Coverage

The test suite includes:

- **Function Testing**: All three main functions with various input scenarios
- **Input Validation**: Tests for invalid inputs (non-square matrices, wrong data types, etc.)
- **Mathematical Properties**: Verification of stationary distribution properties (μP = μ)
- **Edge Cases**: Single state matrices, nearly deterministic matrices, symmetric matrices
- **Error Handling**: Matrix inversion failures, eigenvalue computation errors
- **Paper Examples**: Validation against the example from reference [2]

### Test Categories

1. **TestLocalEntropy**: Tests for the `local_entropy()` function
2. **TestStationaryDistribution**: Tests for the `stationary_distribution()` function
3. **TestTrajectoryEntropy**: Tests for the `trajectory_entropy()` function
4. **TestMathematicalProperties**: Verification of mathematical correctness
5. **TestEdgeCases**: Boundary conditions and special cases

### Example Test Output

```bash
$ pytest -v
=========== test session starts ===========
collected 16 items

test_trajectory_entropy.py::TestLocalEntropy::test_local_entropy_valid_matrix PASSED
test_trajectory_entropy.py::TestLocalEntropy::test_local_entropy_deterministic_state PASSED
test_trajectory_entropy.py::TestStationaryDistribution::test_stationary_distribution_valid_matrix PASSED
...
=========== 16 passed in 0.21s ===========
```

## Requirements

- `numpy`: For numerical computations
- `scipy`: For eigenvalue computations
- `pytest`: For running the test suite

## Error Handling

The code includes robust error handling for:

- Invalid transition matrices (non-square, negative probabilities, rows not summing to 1)
- Numerical issues (matrix inversion failures, eigenvalue computation problems)
- Type validation (ensuring inputs are numpy arrays)

## Mathematical Validation

The test suite verifies important mathematical properties:

- Stationary distributions sum to 1 and satisfy μP = μ
- Entropy values are non-negative and finite
- Deterministic transitions have zero entropy
- Uniform distributions maximize entropy
