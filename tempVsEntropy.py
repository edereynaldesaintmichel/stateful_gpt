import os
import re
import lzma
import matplotlib.pyplot as plt
import pandas as pd

# Provided list of filenames
filenames = [
    "gen_test/base_temp_0dot7.txt",
    "gen_test/base_temp_0dot8.txt",
    "gen_test/base_temp_0dot9.txt",
    "gen_test/base_temp_1dot0.txt",
    "gen_test/base_temp_1dot2.txt",
    "gen_test/base_temp_1dot4.txt",
    "gen_test/base_temp_1dot6.txt",
    "gen_test/base_temp_1dot8.txt",
    "gen_test/base_temp_2dot0.txt",
    "gen_test/base_temp_2dot5.txt",
    "gen_test/base_temp_3dot0.txt",
    "gen_test/stateful_temp_0dot7_recursion.txt",
    "gen_test/stateful_temp_0dot8_recursion.txt",
    "gen_test/stateful_temp_0dot9_recursion.txt",
    "gen_test/stateful_temp_1dot0_recursion.txt",
    "gen_test/stateful_temp_1dot2_recursion.txt",
    "gen_test/stateful_temp_1dot4_recursion.txt",
    "gen_test/stateful_temp_1dot6_recursion.txt",
    "gen_test/stateful_temp_1dot8_recursion.txt",
    "gen_test/stateful_temp_2dot0_recursion.txt",
    "gen_test/stateful_temp_2dot5_recursion.txt",
    "gen_test/stateful_temp_3dot0_recursion.txt",
]

results = []

# Regular expression to parse filenames
# It captures model type (base/stateful) and temperature string (e.g., 0dot7)
pattern = re.compile(r"gen_test/(base|stateful)_temp_([\d+dot\d+]+).*\.txt")

print("Processing files...")

for filepath in filenames:
    match = pattern.match(filepath)
    if not match:
        print(f"Warning: Could not parse filename: {filepath}")
        continue

    model_type = match.group(1)
    temp_str = match.group(2)

    try:
        # Convert temperature string 'XdotY' to float X.Y
        temperature = float(temp_str.replace('dot', '.'))
    except ValueError:
        print(f"Warning: Could not convert temperature '{temp_str}' to float for file: {filepath}")
        continue

    try:
        # Read file content - assuming UTF-8 encoding
        with open(filepath, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # Encode text to bytes for compression and size calculation
        byte_content = text_content.encode('utf-8')
        uncompressed_size = len(byte_content)

        # Handle empty files
        if uncompressed_size == 0:
            print(f"Warning: File is empty: {filepath}")
            entropy_proxy = 1.0 # Or perhaps 0 or NaN, defining 1 means no compression needed/possible
            compressed_size = 0
        else:
            # Compress using lzma
            compressed_content = lzma.compress(byte_content)
            compressed_size = len(compressed_content)

            # Calculate entropy proxy
            if compressed_size > 0:
                entropy_proxy = uncompressed_size / compressed_size
            else:
                # Should ideally not happen for non-empty input with lzma
                print(f"Warning: Compressed size is zero for non-empty file: {filepath}")
                entropy_proxy = float('inf') # Or handle appropriately

        results.append({
            'filepath': filepath,
            'model_type': model_type,
            'temperature': temperature,
            'uncompressed_size': uncompressed_size,
            'compressed_size': compressed_size,
            'entropy_proxy': entropy_proxy
        })
        print(f"Processed: {filepath} -> Temp: {temperature}, Model: {model_type}, Entropy Proxy: {entropy_proxy:.4f}")

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

# Convert results to a Pandas DataFrame for easier manipulation and plotting
df = pd.DataFrame(results)

# --- Plotting ---
if not df.empty:
    plt.style.use('seaborn-v0_8-whitegrid') # Using a nice style
    fig, ax = plt.subplots(figsize=(12, 7))

    # Separate data for base and stateful models
    df_base = df[df['model_type'] == 'base'].sort_values(by='temperature')
    df_stateful = df[df['model_type'] == 'stateful'].sort_values(by='temperature')

    # Plot base model results
    ax.plot(1/df_base['temperature'], df_base['entropy_proxy'], marker='o', linestyle='-', label='Base Model')

    # Plot stateful model results
    ax.plot(1/df_stateful['temperature'], df_stateful['entropy_proxy'], marker='s', linestyle='--', label='Stateful Model')

    # Add labels and title
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Entropy Proxy (Uncompressed Size / Compressed Size)")
    ax.set_title("Temperature vs. Text Entropy Proxy (using LZMA Compression)")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True)

    # Show the plot
    plt.show()

    print("\n--- Summary ---")
    print(df.to_string()) # Print the full dataframe
else:
    print("\nNo data processed, cannot generate plot.")