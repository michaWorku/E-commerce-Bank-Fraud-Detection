# Data

This directory is organized to manage your project's data assets.

## Structure
- `raw/`: Contains original, immutable data sources. These files should ideally not be modified once placed here.
- `processed/`: Contains cleaned, transformed, or feature-engineered data ready for modeling or analysis.

## Best Practices
- **Never commit large data files directly to Git.** Use tools like DVC (Data Version Control) or Git LFS (Large File Storage) for managing large datasets.
- Document data sources, schemas, and transformations in a `data_dictionary.md` or similar file.
- Add `.keep` files to empty `raw/` and `processed/` directories to ensure they are tracked by Git.
