Synthetic Data Generation Pipeline
===============================

Model used: `"meta-llama/Meta-Llama-3.1-8B-Instruct"`

Weights are available at [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct). You may need to accept the model license to access the weights.


## Overview
This directory contains scripts and configurations for generating synthetic data using the Llama 3.1 8B Instruct model. The synthetic data generation process is designed to create high-quality product descriptions by combining the salient information from existing product and tag descriptions.

### Problems with the existing descriptions (label) in the catalog
- Noisy and inconsistent descriptions
- Length varies significantly
- Lack of detail in some descriptions
- Inconsistent formatting and structure
- Most tokenizers of Vision Language Models (VLMs) have a limit of 77 tokens (e.g., CLIP language tokenizer). Descriptions longer than this limit may lead to truncation and loss of important information.

### Why combine product and tag descriptions?
- Product descriptions contain the full context of the product, but they can be noisy and inconsistent.
- Tag descriptions are usually the same length as product descriptions but are more focused on specific attributes or features of the product.
- By combining both descriptions, we can leverage the strengths of each to create a more comprehensive and informative synthetic description, and are also more likely to fit within the token limit of VLMs.

### Synthetic data generation process
1. **Data preparation**:
   - Collect product and tag descriptions from the catalog.
   - Preprocess the descriptions to remove any irrelevant information or formatting issues.
2. **Prompt engineering**:
   - Design prompts that effectively instruct the Llama 3.1 8B Instruct model to generate synthetic descriptions.
   - Prompt used:
        ```text
        [[ ## role ## ]]
        You are a helpful assistant that generates descriptions for grocery products based on their product and tag descriptions.

        [[ ## context ## ]]
        The original descriptions are noisy, lengthy, and inconsistent in format. Your generated descriptions will be used for a multi-product retrieval task, so they should be concise, informative, and relevant to the product. The original descriptions are provided in the following format:
        PRODUCT_DESCRIPTION: <product_description>
        TAG_DESCRIPTION: <tag_description>

        [[ ## task ## ]]
        Generate a concise and informative product description in less than or exactly 77 tokens. The expected output must be a JSON object with a single key \"label\" and the value being the generated description.

        [[ ## expected output ## ]]
        \```json
        {
            \"label\": \"<generated description>\"
        }
        \```

        [[ ## constraints ## ]]
        - Start with \"The product is ...\" in every description.
        - The tag description is more important because it is focused on specific visual attributes or features of the product that are prominent in images.

        [[ ## example ## ]]
        "INPUT:
        PRODUCT_DESCRIPTION: 'The image shows a can of Campbell\'s Condensed Vegetable Beef Soup. The can is predominantly red with white and yellow text. The Campbell\'s logo is at the top, and the product name is prominently displayed in the center. Below the product name, there is an image of the soup inside the can, which includes visible pieces of vegetables and meat.'
        
        TAG_DESCRIPTION: 'The image shows a can of Campbell\'s Condensed Vegetable Beef Soup. The can is predominantly red with white and yellow text. The Campbell\'s logo is at the top in white script, and below it, the words \"CONDENSED SOUP\" are written in smaller white letters. The product name \"Vegetable Beef\" is in bold white letters, and below it, the word \"Soup\" is in a smaller font. There is a visual representation of the soup inside the can, showing a mix of vegetables and beef in a broth.'
        

        OUTPUT:
        \```json
        {
            "label": \"The product is \"Campbell's Beef Vegetable Soup\" which comes in a can that is predominantly red with white and yellow text.\"
        }
        \```
        ```
3. **Model inference**:
   - Use the Llama 3.1 8B Instruct model to generate synthetic descriptions based on the prepared prompts.
   - Ensure that the model adheres to the constraints and formatting specified in the prompt.
4. **Post-processing**:
   - Parse the model's output to extract the generated descriptions.
   - Validate the descriptions to ensure they meet the length and content requirements.
5. **Evaluation**:
   - Compare the synthetic descriptions with the original descriptions to assess improvements in quality and informativeness.
   - In our case, manual inspection and human evaluation were used to assess the quality of the generated descriptions.
6. **Integration**:
   - Use the synthetic descriptions in the multi-product retrieval task instead of the original descriptions.

### Human evaluation results
- A random sample of 250 of the 409 synthetic descriptions (61% of the catalog) was manually evaluated in two independent passes using different random seeds.
- On this sample, 100% of the descriptions met the 77-token limit and retained the key visual attributes (color, shape, brand, size) from the source metadata, with no hallucinations identified. Descriptions consistently started with *"The product is ..."* as instructed.
