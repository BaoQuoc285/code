import re
import string
def preprocessing(text):
    # remove duplicate characters such as đẹppppppp
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)

    # remove punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    # remove '_'
    text = text.replace('_', ' ')

    # remove numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # lower word
    text = text.lower()

    # replace special words
    replace_list = {
        'ô kêi': ' ok ', 'o kê': ' ok ',
        'kh ':' không ', 'kô ':' không ', 'hok ':' không ',
        'kp ': ' không phải ', 'kô ': ' không ', 'ko ': ' không ', 'khong ': ' không ', 'hok ': ' không ',
    }
    for k, v in replace_list.items():
        text = text.replace(k, v)

    # split texts
    texts = text.split()

    # if len(texts) < 5:
    #     return text

    return text.replace("\n","")