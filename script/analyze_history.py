import numpy as np
import os

def analyze_xnes_history(file_path):
    print(f"Analyzing file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        data = np.load(file_path)
        
        # Check keys
        print(f"Keys found: {list(data.keys())}")
        
        if 'center' in data and 'A' in data:
            center = data['center']
            A = data['A']
            
            print("\n--- Shape Analysis ---")
            print(f"Center history shape: {center.shape} (iterations, dimensions)")
            print(f"Matrix A history shape: {A.shape} (iterations, dimensions, dimensions)")
            
            n_iters = center.shape[0]
            dim = center.shape[1]
            
            print(f"\nTotal iterations recorded: {n_iters}")
            print(f"Parameter dimension: {dim}")
            
            print("\n--- Content Preview ---")
            print("Initial Center (first 5 elements):")
            print(center[0, :5])
            
            print("\nFinal Center (first 5 elements):")
            print(center[-1, :5])
            
            print("\nInitial Matrix A diagonal (first 5 elements):")
            print(np.diag(A[0])[:5])
            
            print("\nFinal Matrix A diagonal (first 5 elements):")
            print(np.diag(A[-1])[:5])
            
            # Basic integrity check
            if A.shape[0] != center.shape[0]:
                print("\n[WARNING] Iteration counts mismatch between center and A!")
            else:
                print("\n[OK] Iteration counts match.")
                
            if A.shape[1] != dim or A.shape[2] != dim:
                 print("\n[WARNING] Matrix A dimensions do not match parameter dimension!")
            else:
                 print("\n[OK] Dimensions match.")
                 
        else:
            print("Error: 'center' or 'A' keys missing in the .npz file")
            
    except Exception as e:
        print(f"An error occurred while loading or parsing the file: {e}")

if __name__ == "__main__":
    # Adjust path if necessary
    file_path = "xNES_matrix_history.npz"
    analyze_xnes_history(file_path)


