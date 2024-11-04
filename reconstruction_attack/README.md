# Codes for reconstruction attack
## Dictionary-based attack
To generate a code and embedding dictionary, use the following command format:

```bash
python dictionary_generate.py -i path_dataset -o path_dictionary -n number_of_embedding
```

| Argument     | Description                                                                   |
|--------------|-------------------------------------------------------------------------------|
| `-i (--input)`   | The path to the input dataset                                    |
| `-o (--output)`      |  The output directory where the generated dictionary will be saved (default: "./dictionary")          |
| `-n (--line)`     | The number of lines used for embedding (default: 1)      |


To conduct a dictionary-based code reconstruction attack, use the following command format:

```bash
python dictionary_based.py -i path_dictionary 
```
| Argument     | Description                                                                   |
|--------------|-------------------------------------------------------------------------------|
| `-i (--input)`  | The path to the dictionary (default: "./dictionary")          |

## RNN-based attack and LLM-based attack
For the RNN-based attack and LLN-based attack, we utilize the source codes from [GEIA, ACL 2023](https://github.com/HKUST-KnowComp/GEIA).
