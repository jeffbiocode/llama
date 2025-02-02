from typing import List, Optional

import fire

from llama import Llama, Dialog

if __name__ == "__main__":
    # Initialize the Llama generator outside the loop
    generator = Llama.build(
        ckpt_dir="llama-2-7b-chat/",
        tokenizer_path="tokenizer.model",
        max_seq_len=512,
        max_batch_size=6
    )
    # Collect user input for system and user roles
    system_input = input("Enter system message:")
    while True:        
        user_input = input("Enter user message (or type 'exit' to quit or 'system' to enter new system message): ")

        # Check for the exit condition
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "system":
            system_input = input("Enter system message:")
            user_input = input("Enter user message:")

        # Wrap the input in a dialog format
        dialog = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": user_input}
        ]

        # Process with the generator
        results = generator.chat_completion(
            [dialog],
            max_gen_len=None, 
            temperature=0.6,
            top_p=0.9
        )

        # Print the results to the screen
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(f"{results[0]['generation']['content']}\n")
        print("\n==================================\n")
