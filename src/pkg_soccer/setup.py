from setuptools import setup

package_name = 'pkg_soccer'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    scripts=['start_match.sh'],
    package_data={package_name: ['policy_stage2_actor.npz',
                                 'policy_duel_attacker_actor.npz',
                                 'policy_duel_defender_actor.npz',
                                 'policy_2v2_shared_actor.npz',
                                 'policy_team_attacker_actor.npz',
                                 'policy_2v2_attack_actor.npz',
                                 'policy_2v2_assist_actor.npz']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mi',
    maintainer_email='mi@todo.todo',
    description='Closed-loop CyberDog soccer agent using VRPN motion capture.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'soccer = pkg_soccer.soccer:main',
        'attacker = pkg_soccer.soccer:attacker_main',
        'record_motion = pkg_soccer.record_motion:main',
    ]},
)
