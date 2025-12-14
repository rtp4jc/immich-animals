# Immich Animals - Project Status

## Coding Guidelines

### Code Style & Architecture

- **Protocol-Based Design**: Use Python protocols for dependency injection and modularity
- **Generic Architecture**: Design for extensibility (animals, not just dogs; multiple model types)
- **Elegant Naming**: Use fun, memorable names with clear progression (AmbidextrousAxolotl → BrilliantBadger)

### Documentation & Comments

- **Focus on Why**: Document design decisions, trade-offs, and requirements, not implementation details

### Error Handling & Robustness

- **Don't assume**: Don't assume files or models will be present. Check before or catch exceptions when you can recover from a missing file or directory
- **Preserve Original Features**: When refactoring, maintain existing functionality (progress bars, visualizations, etc.)
- **Field Name Consistency**: Update all related code when changing data structures
- **Progress Indicators**: Use `tqdm` progress bars for any process that could take more than 5 seconds and can be broken down into small work items

### Testing & Validation

- **Benchmark-Driven**: Use quantitative metrics to validate design decisions
- **Comparative Analysis**: Test multiple approaches (with/without keypoints) side-by-side
- **Visual Validation**: Provide visualization tools for understanding system behavior

### Integration Patterns

- **Refactor for Reuse**: Extract common visualization/utility code to shared modules
- **One-line Commits**: Use concise single-line commit messages for small changes

## Environment

- **Platform**: write code to work in WSL (Linux) or Windows
- **Python**: Use `python312` conda environment for all scripts
- **Note**: Some scripts may have Windows-specific paths/issues - originally developed for native Windows
- **Data Files**: Do not directly read files in `/data` directory (too large). Use head/tail or jq for parsing

### Immich Container Management

```bash
# Build and run the custom Immich ML container
cd immich-clone/machine-learning
docker build -f Dockerfile.dogs -t immich-ml-dogs .
docker run -d --name immich-ml-dogs -p 3003:3003 immich-ml-dogs

# Container operations
docker stop immich-ml-dogs
docker start immich-ml-dogs
docker logs immich-ml-dogs --tail 20

# Test API endpoint
curl -X POST http://localhost:3003/predict \
  -F "image=@/path/to/dog/image.jpg" \
  -F 'entries={"dog-identification":{"detection":{"modelName":"dog_detector"},"recognition":{"modelName":"dog_embedder_direct"}}}'
```

### Script Output Handling

When running scripts with large outputs, save to file and read selectively:

```bash
# Save output to file
conda run -n python312 python scripts/08_validate_embeddings.py > outputs/scripts/validation_output.txt 2>&1

# Read only head/tail of output
head -20 outputs/scripts/validation_output.txt
tail -20 outputs/scripts/validation_output.txt
```

## Lessons Learned

### **Dataset Handling:**

- **Negative samples are normal** - YOLO datasets include images without annotations
- **Path mapping is critical** - `images/` ↔ `labels/` conversion must be exact
- **Filter for meaningful visualization** - Show only samples with actual annotations

### **Code Architecture Validation:**

- **Import patterns consistent** - PROJECT_ROOT + sys.path.append pattern works reliably

## Adding new dependencies

If a new dependency is needed, follow the below steps:

1. Try to install the package with `conda install` 
2. If it doesn't appear to be registered in conda use `pip install`
3. ONLY AFTER installing, add the version that was installed to the `requirements.txt` file