from setuptools import setup

package_name = 'pkg_1106'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mi',
    maintainer_email='mi@todo.todo',
    description='ROS 2 camera learning package.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'camera_rgb = pkg_1106.camera_rgb:main',
        ],
    },
)
