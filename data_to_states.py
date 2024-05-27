import numpy as np

size_of_the_chain = 5

# Load the data from the file
with open('datasets/Eigenvectors_results_10_feast.dat', 'r') as f:
    lines = f.readlines()

# Split the lines into eigenvectors
eigenvectors = []
temp_vector = []
for line in lines[1:]:
    if line.strip():  # if line is not empty
        temp_vector.extend(line.strip().split())
    else:  # if line is empty
        eigenvectors.append(temp_vector)
        temp_vector = []

# Don't forget to add the last eigenvector
if temp_vector:
    eigenvectors.append(temp_vector)

# Convert the eigenvectors to a numpy array and reshape them
vectors = np.array(eigenvectors, dtype=float)
#breakpoint()
with open('training_states_10/dictionary.txt', 'w') as dict_file:
    for j in range(vectors.shape[0]):
        
        tensor = vectors[j].reshape(tuple(2 for _ in range(size_of_the_chain)))

        # Save the tensor to a file
        np.save(f'./training_states_10/train/{size_of_the_chain}_tensor_state_{j}.npy', tensor)

        # Write to the dictionary file
        dict_file.write(f'{j+1} {j+1}\n')
    