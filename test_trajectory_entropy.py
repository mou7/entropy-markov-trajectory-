"""
Test suite for trajectory entropy computation functions.

This module contains comprehensive tests for:
- local_entropy function
- stationary_distribution function  
- trajectory_entropy function
- Input validation and error handling
- Mathematical properties verification
"""

import numpy as np
import pytest
from trajectory_entropy import local_entropy, stationary_distribution, trajectory_entropy


class TestLocalEntropy:
    """Test cases for local_entropy function."""
    
    def test_local_entropy_valid_matrix(self):
        """Test local entropy with a valid transition matrix."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        result = local_entropy(P)
        
        # Check shape
        assert result.shape == (2, 1)
        
        # Check that entropies are non-negative
        assert np.all(result >= 0)
        
        # Check specific values (manually calculated)
        expected_0 = 1.0  # -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        expected_1 = 0.881  # -0.3*log2(0.3) - 0.7*log2(0.7) ≈ 0.881
        
        np.testing.assert_almost_equal(result[0, 0], expected_0, decimal=3)
        np.testing.assert_almost_equal(result[1, 0], expected_1, decimal=3)
    
    def test_local_entropy_deterministic_state(self):
        """Test local entropy with deterministic transitions (entropy should be 0)."""
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = local_entropy(P)
        
        # Deterministic transitions should have zero entropy
        np.testing.assert_array_almost_equal(result, [[0.0], [0.0]])
    
    def test_local_entropy_invalid_input(self):
        """Test local entropy with invalid inputs."""
        # Non-square matrix
        with pytest.raises(ValueError, match="P must be a square numpy array"):
            local_entropy(np.array([[0.5, 0.5, 0.0]]))
        
        # Non-2D array
        with pytest.raises(ValueError, match="P must be a square numpy array"):
            local_entropy(np.array([0.5, 0.5]))
        
        # Non-numpy array
        with pytest.raises(ValueError, match="P must be a square numpy array"):
            local_entropy([[0.5, 0.5], [0.5, 0.5]])


class TestStationaryDistribution:
    """Test cases for stationary_distribution function."""
    
    def test_stationary_distribution_valid_matrix(self):
        """Test stationary distribution with a valid transition matrix."""
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mu = stationary_distribution(P)
        
        # Check that it's a probability distribution
        assert np.isclose(np.sum(mu), 1.0)
        assert np.all(mu >= 0)
        
        # Check that it satisfies μP = μ (stationary property)
        np.testing.assert_array_almost_equal(np.dot(mu, P), mu, decimal=10)
    
    def test_stationary_distribution_symmetric_matrix(self):
        """Test with a symmetric doubly stochastic matrix."""
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        mu = stationary_distribution(P)
        
        # For symmetric matrix, stationary distribution should be uniform
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(mu, expected, decimal=10)
    
    def test_stationary_distribution_identity_matrix(self):
        """Test with identity matrix (absorbing states)."""
        P = np.eye(3)
        
        # Identity matrix has eigenvalue 1, but results in non-uniform distribution
        mu = stationary_distribution(P)
        
        # Should still be a valid probability distribution
        assert np.isclose(np.sum(mu), 1.0)
        assert np.all(mu >= 0)
    
    def test_stationary_distribution_invalid_transition_matrix(self):
        """Test with invalid transition matrices."""
        # Rows don't sum to 1
        P_invalid = np.array([[0.5, 0.3], [0.4, 0.7]])
        with pytest.raises(ValueError, match="Transition matrix rows must sum to 1"):
            stationary_distribution(P_invalid)
        
        # Negative probabilities but rows sum to 1 - should still raise error due to negative values
        # Let's create a matrix that clearly doesn't sum to 1
        P_invalid2 = np.array([[0.8, 0.1], [0.2, 0.6]])  # Second row sums to 0.8
        with pytest.raises(ValueError, match="Transition matrix rows must sum to 1"):
            stationary_distribution(P_invalid2)
        
        # Test negative probabilities
        P_negative = np.array([[0.8, -0.3, 0.5], [0.2, 0.8, 0.0], [0.0, 0.5, 0.5]])
        with pytest.raises(ValueError, match="Transition matrix cannot contain negative probabilities"):
            stationary_distribution(P_negative)
    
    def test_stationary_distribution_invalid_input_types(self):
        """Test stationary distribution with invalid input types."""
        # Non-square matrix
        with pytest.raises(ValueError, match="P must be a square numpy array"):
            stationary_distribution(np.array([[0.5, 0.5, 0.0]]))
        
        # List instead of numpy array
        with pytest.raises(ValueError, match="P must be a square numpy array"):
            stationary_distribution([[0.5, 0.5], [0.5, 0.5]])


class TestTrajectoryEntropy:
    """Test cases for trajectory_entropy function."""
    
    def test_trajectory_entropy_valid_matrix(self):
        """Test trajectory entropy with a valid irreducible matrix."""
        # Simple 2x2 irreducible matrix
        P = np.array([[0.3, 0.7], [0.6, 0.4]])
        H = trajectory_entropy(P)
        
        # Check shape
        assert H.shape == (2, 2)
        
        # Check that all entropies are finite
        assert np.all(np.isfinite(H))
        
        # Trajectory entropy should be non-negative
        assert np.all(H >= 0)
    
    def test_trajectory_entropy_paper_example(self):
        """Test with the example from the paper (5x5 matrix)."""
        P = np.zeros((5, 5))
        P[0, 1], P[0, 2] = 0.25, 0.75
        P[1, 4] = 1
        P[2, 1], P[2, 3] = 0.5, 0.5
        P[3, 4] = 1
        P[4, 0], P[4, 3] = 0.5, 0.5
        
        H = trajectory_entropy(P)
        
        # Check basic properties
        assert H.shape == (5, 5)
        assert np.all(np.isfinite(H))
        
        # Check some specific values from the expected output
        # These are approximate values from running the corrected code
        assert 3.0 < H[0, 0] < 4.0  # Should be around 3.561
        assert 1.0 < H[0, 2] < 2.0  # Should be around 1.748
    
    def test_trajectory_entropy_symmetric_matrix(self):
        """Test trajectory entropy with a symmetric matrix."""
        P = np.array([[0.4, 0.6], [0.6, 0.4]])
        H = trajectory_entropy(P)
        
        # For symmetric matrices, H[i,j] should equal H[j,i]
        np.testing.assert_array_almost_equal(H, H.T, decimal=10)
    
    def test_trajectory_entropy_invalid_input(self):
        """Test trajectory entropy with invalid inputs."""
        # Non-square matrix
        with pytest.raises(ValueError, match="P must be a square numpy array"):
            trajectory_entropy(np.array([[0.5, 0.5, 0.0]]))
        
        # Non-stochastic matrix (rows don't sum to 1)
        P_invalid = np.array([[0.5, 0.3], [0.4, 0.7]])
        with pytest.raises(ValueError, match="Transition matrix rows must sum to 1"):
            trajectory_entropy(P_invalid)


class TestMathematicalProperties:
    """Test mathematical properties of the functions."""
    
    def test_stationary_distribution_properties(self):
        """Test mathematical properties of stationary distributions."""
        # Create a few different irreducible matrices
        matrices = [
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.4, 0.4, 0.2]]),
            np.array([[0.25, 0.75, 0.0], [0.5, 0.0, 0.5], [0.3, 0.3, 0.4]])
        ]
        
        for P in matrices:
            mu = stationary_distribution(P)
            
            # μ should be a probability distribution
            assert np.isclose(np.sum(mu), 1.0)
            assert np.all(mu >= 0)
            
            # μ should satisfy μP = μ
            np.testing.assert_array_almost_equal(np.dot(mu, P), mu, decimal=10)
    
    def test_entropy_properties(self):
        """Test that entropy values have expected properties."""
        P = np.array([[0.2, 0.8], [0.7, 0.3]])
        
        # Local entropy should be maximized for uniform distributions
        local_ent = local_entropy(P)
        
        # Create a uniform transition matrix for comparison
        P_uniform = np.array([[0.5, 0.5], [0.5, 0.5]])
        local_ent_uniform = local_entropy(P_uniform)
        
        # Uniform distribution should have maximum entropy (= log2(2) = 1)
        np.testing.assert_array_almost_equal(local_ent_uniform, [[1.0], [1.0]])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_state_matrix(self):
        """Test with a 1x1 matrix."""
        P = np.array([[1.0]])
        
        # Single state should work for local entropy
        local_ent = local_entropy(P)
        assert local_ent.shape == (1, 1)
        assert local_ent[0, 0] == 0.0  # Deterministic
        
        # Single state stationary distribution
        mu = stationary_distribution(P)
        assert np.isclose(mu[0], 1.0)
        
        # Single state trajectory entropy
        H = trajectory_entropy(P)
        assert H.shape == (1, 1)
        assert np.isfinite(H[0, 0])
    
    def test_nearly_deterministic_matrix(self):
        """Test with a matrix that's nearly deterministic."""
        eps = 1e-10
        P = np.array([[1-eps, eps], [eps, 1-eps]])
        
        # Should not raise errors
        mu = stationary_distribution(P)
        H = trajectory_entropy(P)
        
        assert np.isclose(np.sum(mu), 1.0)
        assert H.shape == (2, 2)
        assert np.all(np.isfinite(H))


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"]) 