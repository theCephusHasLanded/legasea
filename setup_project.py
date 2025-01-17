import os

# Define the project structure
structure = {
    "legasea": {
        "models": ["aerial_detection"],
        "data": ["preprocessing", "postprocessing"],
        "api": ["cloud_services"],
        "frontend": ["property_dashboard"],
        "utils": ["image_processing"],
        "tests": [],
        "docs": [],
    }
}

# Create the directories and README.md
def create_project_structure(base_dir, structure):
    for parent, subdirs in structure.items():
        parent_path = os.path.join(base_dir, parent)
        os.makedirs(parent_path, exist_ok=True)

        # Create subdirectories
        for subdir in subdirs:
            subdir_path = os.path.join(parent_path, subdir)
            os.makedirs(subdir_path, exist_ok=True)

        # Create a README.md file in the root
        if parent == "legasea":
            with open(os.path.join(parent_path, "README.md"), "w") as readme:
                readme.write("# Legasea Project\n\nThis project contains tools for aerial disaster detection and visualization.")

# Execute the script
base_directory = os.getcwd()  # Current working directory
create_project_structure(base_directory, structure)
print("Project structure created!")
