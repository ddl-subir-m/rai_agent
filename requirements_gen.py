import subprocess
import sys


def create_requirements_file(specific_versions=False):
    packages = [
        "autogen",
        "openai",
        "python-dotenv",
        "faiss-cpu",
        "torch",
        "transformers",
        "sentence-transformers"
    ]

    if specific_versions:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
            installed_packages = result.stdout.split('\n')

            with open('requirements.txt', 'w') as f:
                for package in installed_packages:
                    if any(p in package for p in packages):
                        f.write(f"{package}\n")
            print("requirements.txt created with specific versions.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
    else:
        with open('requirements.txt', 'w') as f:
            for package in packages:
                f.write(f"{package}\n")
        print("requirements.txt created without version specifications.")


if __name__ == "__main__":
    user_input = input("Do you want to include specific versions? (y/n): ").lower()
    create_requirements_file(user_input == 'y')