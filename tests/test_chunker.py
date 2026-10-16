from src.chunker import chunk_python, chunk_text


def test_chunk_python():
    content = """def hello():
    print("hello")


def goodbye():
    print("goodbye")
"""

    chunks = chunk_python(
        content,
        "test.py",
        2000,
    )

    assert len(chunks) == 2

    for chunk in chunks:
        assert len(chunk.text) <= 2000
        assert (
            content[
                chunk.first_character_index:
                chunk.last_character_index
            ]
            == chunk.text
        )


def test_chunk_python_too_large():
    content = "def hello():\n" + "    print('hello')\n" * 100

    chunks = chunk_python(
        content,
        "test.py",
        100,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) <= 100
        assert (
            content[
                chunk.first_character_index:
                chunk.last_character_index
            ]
            == chunk.text
        )


def test_chunk_text():
    content = """First paragraph.

Second paragraph.

Third paragraph.
"""

    chunks = chunk_text(
        content,
        "test.md",
        100,
    )

    assert len(chunks) == 1

    for chunk in chunks:
        assert (
            content[
                chunk.first_character_index:
                chunk.last_character_index
            ]
            == chunk.text
        )


def test_invalid_python():
    content = "def hello(:"

    chunks = chunk_python(
        content,
        "test.py",
        2000,
    )

    assert chunks == []

def test_chunk_python_unicode():
    content = '''def hello():
    message = "café été"
    print(message)

def goodbye():
    print("goodbye")
'''

    chunks = chunk_python(content, "test.py", 2000)

    assert len(chunks) == 2

    for chunk in chunks:
        assert content[
            chunk.first_character_index:chunk.last_character_index
        ] == chunk.text

def test_chunk_text_too_large():
    content = "A" * 250

    chunks = chunk_text(content, "test.md", 100)

    assert len(chunks) == 3

    for chunk in chunks:
        assert len(chunk.text) <= 100
        assert content[
            chunk.first_character_index:chunk.last_character_index
        ] == chunk.text


def test_chunk_python_class_method_signature():
    content = '''class LLM:
    def __init__(self):
        """Initialize the LLM."""
        trust_remote_code: bool = False
'''

    chunks = chunk_python(content, "test.py", 2000)

    assert len(chunks) == 1
    assert "def __init__" in chunks[0].text
    assert "trust_remote_code: bool = False" in chunks[0].text



def test_chunk_python_large_class_method():
    content = (
        "class LLM:\n"
        "    def __init__(self):\n"
        '        """Initialize the LLM."""\n'
        "        trust_remote_code: bool = False\n"
        + "        value = 1\n" * 30
    )

    chunks = chunk_python(content, "test.py", 100)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) <= 100
        assert (
            content[
                chunk.first_character_index:
                chunk.last_character_index
            ] == chunk.text
        )

def test_chunk_python_keeps_method_signature_together():
    content = (
        "class LLM:\n"
        "    def __init__(self, model: str):\n"
        "        trust_remote_code: bool = False\n"
        + "        value = 1\n" * 30
    )

    chunks = chunk_python(content, "test.py", 100)

    assert len(chunks) > 1

    assert any(
        "def __init__(self, model: str)" in chunk.text
        and "trust_remote_code: bool = False" in chunk.text
        for chunk in chunks
    )