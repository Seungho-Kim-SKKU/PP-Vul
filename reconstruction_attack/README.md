# Codes for reconstruction attack

To generate code and embedding dictionary, use the following command format:

```bash
python dictionary_generate.py -i path_dataset -o path_dictionary -n number_of_embedding
```

| Argument     | Description                                                                   |
|--------------|-------------------------------------------------------------------------------|
| `-i` (--input)   | The path to the input dataset                                    |
| `-o` (--output)      |  The output directory where the generated dictionary will be saved (defualt "./dictionary")          |
| `-n` (--line)     | The number of lines used for embedding (default 1)      |


To conduct dictionary_based code reconstruction attack, use the following command format:

```bash
python dictionary_generate.py -i path_dataset -o path_dictionary -n number_of_embedding
```
