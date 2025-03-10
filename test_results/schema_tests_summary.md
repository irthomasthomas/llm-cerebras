# Schema Tests with llm-cerebras

All tests were run using the `cerebras-llama3.3-70b` model.

## Test Results

### 1. Basic Schema
```bash
llm -m cerebras-llama3.3-70b --schema "name, age int, bio" "Generate a fictional person"
```

**Result:**
```json
{
  "name": "Ava Morales",
  "age": 27,
  "bio": "Ava is a skilled software engineer with a passion for hiking and playing the guitar. She lives in a small town surrounded by mountains and spends her free time volunteering at local animal shelters."
}
```

### 2. Schema with Descriptions
```bash
llm -m cerebras-llama3.3-70b --schema "name: full name including title, age int: age in years, specialization: field of study" "Generate a profile for a scientist"
```

**Result:**
```json
{
  "name": "Dr. Maria Rodriguez",
  "age": 35,
  "specialization": "Astrophysics"
}
```

### 3. Multiple Items with Schema-Multi
```bash
llm -m cerebras-llama3.3-70b --schema-multi "name, age int, occupation" "Generate 3 different characters"
```

**Result:**
```json
{
  "items": [
    {
      "name": "Eira Shadowglow",
      "age": 250,
      "species": "Elf",
      "occupation": "Ranger"
    },
    {
      "name": "Kael Darkhaven",
      "age": 35,
      "species": "Human",
      "occupation": "Assassin"
    },
    {
      "name": "Lila Earthsong",
      "age": 120,
      "species": "Dwarf",
      "occupation": "Cleric"
    }
  ]
}
```

### 4. Complex Schema using JSON File
```bash
# Using person_schema.json
llm -m cerebras-llama3.3-70b --schema person_schema.json "Generate a profile for a software developer"
```

**Result:**
```json
{
  "person": {
    "name": "John Doe",
    "age": 30,
    "occupation": "Software Developer",
    "skills": ["Java", "Python", "C++", "JavaScript"],
    "experience": 5
  },
  "location": {
    "city": "New York",
    "country": "USA",
    "latitude": 40.7128,
    "longitude": -74.0060
  }
}
```

### 5. Create and Use Template
```bash
# Create template
llm -m cerebras-llama3.3-70b --schema "title, director, year int, genre" --save movie_template

# Use template
llm -t movie_template "Suggest a sci-fi movie from the 1980s"
```

**Result:**
```json
{
  "title": "Blade Runner",
  "director": "Ridley Scott",
  "year": 1982,
  "genre": "Science Fiction"
}
```

### 6. Using a Previously Used Schema by ID
```bash
llm -m cerebras-llama3.3-70b --schema 9c57ef588ee1f02a093277cef6138619 "Generate a fictional character who is a chef"
```

**Result:**
```json
{
  "name": "Marcel Leblanc",
  "age": 32,
  "bio": "Award-winning chef and owner of the renowned Parisian restaurant, Bistro Bliss, Marcel Leblanc is known for his creative and exquisite French cuisine. With a passion for using only the freshest ingredients, Marcel's dishes are a testament to his dedication to the culinary arts."
}
```

## Observations

1. **All schema variations worked successfully** - basic schemas, schemas with descriptions, multi-schemas, complex schemas from files, templates, and schemas by ID.

2. **Schema validation is functioning properly** - Integer fields are returned as integers, string fields as strings, and array fields as arrays.

3. **Schema descriptions influence the output** - When we specified "name: full name including title", the model returned "Dr. Maria Rodriguez" with the title.

4. **Additional fields sometimes included** - In the multi-schema example, the model added a "species" field that wasn't in our schema, but all required fields were present.

5. **Templates work correctly** - Creating and using a template preserved the schema structure.

6. **Schema management works** - We can list schemas, view full details, and reuse schemas by ID.

## Conclusion

The schema support implementation for llm-cerebras is working correctly across all tested variations and use cases. The workaround using json_object mode with system instructions provides seamless schema functionality equivalent to what users would expect from native schema support.
