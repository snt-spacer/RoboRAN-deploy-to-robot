from setuptools import setup
import os
from glob import glob

package_name = "goal_generator"

def generate_data_files():
    pairs = []
    for path, dirs, files in os.walk("config"):
        if files:
            target_dir = os.path.join("share", package_name, path)
            data_files = [os.path.join(path, file) for file in files]
            pairs.append((target_dir, data_files))
    return pairs

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/**/*launch.[pxy][yma]", recursive=True)),
    ] + generate_data_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="antoine",
    maintainer_email="antoine.richard@uni.lu",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "goal_generator_node = goal_generator.goal_generator_node:main",
        ],
    },
)
