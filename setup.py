# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

# Add the plugin dependencies here
requirements = ['numpy', 'matplotlib', 'pysteps']

# Add the packages needed to build the package.
setup_requirements = ['pytest-runner']

test_requirements = ['pytest>=3']

entry_label = 'pysteps.plugins.' + 'diagnostic'

# It woudld be even better to read the functions from the plugin module.
# We could add multiple functions in the entry_points.
# Is that possible?
# e.g. like plugin_functions = [attr for attr in dir(importlib.import_module(cookiecutter.project_slug.cookiecutter.plugin_type.cookiecutter.plugin_name)) if attr.startswith("import_" if "importer" in cookiecutter.plugin_type else cookiecutter.plugin_type+"_")]
# Then loop over the functions to set all entry_points.
entry = {
    entry_label: [
        'diagnostic_prtype=pysteps_diagnostic_prtype.diagnostic.diagnostic_prtype:diagnostic_prtype',
        'calculate_precip_type=pysteps_diagnostic_prtype.diagnostic.diagnostic_prtype:calculate_precip_type',
        'get_reprojected_indexes=pysteps_diagnostic_prtype.diagnostic.diagnostic_prtype:get_reprojected_indexes',
        'grid_interpolation=pysteps_diagnostic_prtype.diagnostic.diagnostic_prtype:grid_interpolation',
        'create_timestamp_indexing=pysteps_diagnostic_prtype.diagnostic.diagnostic_prtype:create_timestamp_indexing',
        'generate_interpolations=pysteps_diagnostic_prtype.diagnostic.diagnostic_prtype:generate_interpolations',
    ]
}

setup(
    author="PySTEPS_developers",
    author_email='your@email.com',
    python_requires='>=3.10',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.12'
    ],
    description="Pysteps plugin for calculating the precipitation type of hydrometeors.",
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    test_suite='tests',
    tests_require=test_requirements,
    include_package_data=True,
    keywords=['pysteps_diagnostic_prtype', 'pysteps' , 'plugin', 'diagnostic'],
    name='pysteps-diagnostic-prtype',
    packages=find_packages(),
    setup_requires=setup_requirements,
    # Entry points
    # ~~~~~~~~~~~~
    #
    # This is the most important part of the plugin setup script.
    # Entry points are a mechanism for an installed python distribution to advertise
    # some of the components installed (packages, modules, and scripts) to other
    # applications (in our case, pysteps).
    # https://packaging.python.org/specifications/entry-points/
    #
    # An entry point is defined by three properties:
    # - The group that an entry point belongs indicate the kind of functionality that
    #   provides. For the pysteps importers use the "pysteps.plugins.importers" group.
    #   For the pysteps diagnostics use the "pysteps.plugins.diagnostics" group.
    # - The unique name that is used to identify this entry point in the
    #   "pysteps.plugins.importers" group.
    # - A reference to a Python object. For the pysteps importers, the object should
    #   point to a importer function, and should have the following form:
    #   package_name.module:function.
    # The setup script uses a dictionary mapping the entry point group names to a list
    # of strings defining the importers provided by this package (our plugin).
    # The general form of the entry points dictionary is:
    # entry_points={
    #     "group_name": [
    #         "entry_point_name=package_name.module:function",
    #         "entry_point_name=package_name.module:function2",
    #     ]
    # },
    entry_points = entry,
    version='0.1.0',
    zip_safe=False,
)
