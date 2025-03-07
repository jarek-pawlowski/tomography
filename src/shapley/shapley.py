import numpy as np

################################################################################

def partial_predict_proba(model, x, S, X_bg, class_idx=1):
    """
    Approximate the model prediction f_S(x), where only features in S are 'fixed' from x
    and the other features (not in S) are taken from background samples X_bg.

    Parameters
    ----------
    model : model
        Trained binary classifier model providing predict_proba method (scikit-learn).
    x : np.ndarray, shape (M,)
        Single input instance for which we want the partial prediction.
    S : list or np.ndarray
        Indices of features to keep from x.
    X_bg : np.ndarray, shape (B, M)
        Background dataset with B samples and M features.
    class_idx : int
        Which class probability to return (1 for positive class, typically).

    Returns
    -------
    float
        Mean predicted probability for the positive class under the partial knowledge scenario.
    """
    
    # Build an array of hybrid samples
    # For each background sample z, we keep x[S] and use z for the other features
    B = X_bg.shape[0]  # Number of hybrid samples
    M = x.shape[0]     # Number of features

    # Collect all the hybrid samples
    hybrid_samples = np.zeros((B, M))
    for i in range(B):
        # Create a hybrid sample using the sampe x and a background sample
        hybrid_samples[i]    = X_bg[i]  # Copy the entire background sample
        hybrid_samples[i, S] = x[S]     # Overwrite the features in S with x's features

    # Get predicted probabilities from the model
    proba_bg = model.predict_proba(hybrid_samples)   # shape: (B, 2) [prob class=0, prob class=1]

    # Return the average probability of the chosen class
    return np.mean(proba_bg[:, class_idx])

################################################################################

def approximate_shap_values(model, x, X_bg, n_permutations=50, class_idx=1, random_state=42):
    """
    Approximate Shapley values for a trained model using permutation sampling.

    Parameters
    ----------
    model : model
        A binary classifier model providing predict_proba method (e.g., using scikit-learn).
    x : np.ndarray, shape (M,)
        Single data sample for which we compute the Shapley values.
    X_bg : np.ndarray, shape (B, M)
        Background dataset used for marginalization.
    n_permutations : int
        Number of random permutations to sample for each feature.
    class_idx : int
        Class index for which to compute probabilities (1 = positive class).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    shap_values : np.ndarray, shape (M,)
        The approximate Shapley value for each of the M features in x.
    """

    # Initialize random numbers generator
    rng = np.random.default_rng(random_state)
    
    M = x.shape[0]              # Number of features
    shap_values = np.zeros(M)   # Initialized list of Shapley values

    # Calculate Shapley contribution for each feature
    for i in range(M):
        diff_sum = 0.0
        
        # Sample multiple random permutations
        for _ in range(n_permutations):
            perm = rng.permutation(M)
            # Find where feature i appears in perm
            pos_i = np.where(perm == i)[0][0]
            # Set S: all features that appear before i
            S = perm[:pos_i]

            # Compute f_S(x) - without feature i
            fS = partial_predict_proba(model, x, S, X_bg, class_idx=class_idx)

            # Compute f_{S union i}(x) - with feature i
            S_plus_i = np.concatenate([S, [i]])
            fS_plus_i = partial_predict_proba(model, x, S_plus_i, X_bg, class_idx=class_idx)

            # Calculate marginal contribution
            diff_sum += (fS_plus_i - fS)

        # The Shapley value for feature i: the average difference over permutations
        shap_values[i] = diff_sum / n_permutations

    return shap_values

################################################################################
