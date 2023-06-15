def read_config_file(file_path):
    config = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                if value.isdigit():
                    config[key] = int(value)
                elif value.lower() == 'true':
                    config[key] = True
                elif value.lower() == 'false':
                    config[key] = False
                else:
                    config[key] = value
    return config

if __name__ == '__main__':
    # 設定ファイルのパス
    config_file_path = '../tests/config.txt'

    # 設定ファイルを読み込む
    config = read_config_file(config_file_path)

    # 読み込んだ設定をプログラム内で利用する例
    hidden_num = config['HIDDEN_NUM']
    input_num = config['INPUT_NUM']
    output_num = config['OUTPUT_NUM']
    connection_num = config['CONNECTION_NUM']
    has_internal_state = config['HAS_INTERNAL_STATE']

    # 利用例として、読み込んだ設定を出力してみる
    print("Hidden neurons:", hidden_num)
    print("Input neurons:", input_num)
    print("Output neurons:", output_num)
    print("Number of connections:", connection_num)
    print("Has internal state:", has_internal_state)