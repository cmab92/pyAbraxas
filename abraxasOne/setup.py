from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='abraxasOne',
    version='0.1',
    description='...',
    url='http://github.com/cmab92/pyAbraxas',
    author='cb',
    author_email='bonenbch@hs-weingarten.de',
    license='...',
    packages=['abraxasOne'],
    install_requires=[
        'numpy',
        'csv',
        'datetime',
        'serial',
    ],
    zip_safe=False)