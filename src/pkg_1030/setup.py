from setuptools import setup

package_name = 'pkg_1030'

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
    description='ROS 2 service learning package.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'service_motion = pkg_1030.service_motion:main',
            'face_enroll = pkg_1030.face_enroll:main',
            'face_recognize_led = pkg_1030.face_recognize_led:main',
        ],
    },
)
