import os
from glob import glob
from setuptools import setup

package_name = "fp_inference"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]')),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="antoine",
    maintainer_email="antoine.richard@uni.lu",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rl_task_node = fp_inference.rl_task_node:main",
            "rl_task_node_v2 = fp_inference.rl_task_node_v2:main",
        ],
    },
)
