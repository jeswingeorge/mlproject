from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e .' # creating the variable for the editable install

# This function will return a list of requirements
def get_requirements(file_path: str) -> list:
    """"
    This function will return a list of requirements
    """
    requirments = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        ## removing the \n from the end of each line
        requirments = [req.replace('\n', '') for req in requirments]

        ## removing the -e . from the list
        if HYPHEN_E_DOT in requirments:
            requirments.remove(HYPHEN_E_DOT)
    
    return requirments

setup(
    name='ml_project',
    version='0.1.0',
    author='Jeswin George', # Replace with your name 
    author_email='socialmedia2jeswin@gmail.com',
    description='A machine learning project template',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
    # install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'jupyterlab', 'pytest' ],

)
