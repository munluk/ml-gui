import json


class ConfigProvider(object):
    @staticmethod
    def get_classifiers():
        """
        returns a list of strings, containing the names of the classifiers as plain python strings
        :return:
        """
        with open('src/resources/classifier.config') as f:
            content = f.read()
        config = json.loads(content, encoding='utf-8')
        return config['classifier']

    @staticmethod
    def get_mlp_config():
        """
        returns a list of strings, containing the names of the classifiers as plain python strings
        :return:
        """
        with open('src/resources/classifier.config') as f:
            content = f.read()
        config = json.loads(content, encoding='utf-8')
        return config['mlp_config']

if __name__== '__main__':
    print ConfigProvider.get_classifiers()