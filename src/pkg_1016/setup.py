from setuptools import setup

package_name = 'pkg_1016'

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
    description='Ultrasonic avoidance and laser obstacle observation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'helloworld=pkg_1016.node_helloworld:main',
            'move_the_dog=pkg_1016.move:main',
            'ultrasonic_avoidance=pkg_1016.ultrasonic_avoidance:main',
            'lidar_observer=pkg_1016.lidar_observer:main',
        ],
    },
)
