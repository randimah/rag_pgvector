all: setup run

setup:
	$ pip install -r requirements.txt

run:
	$ python -m data_vectorizer;
	$ python -m rag