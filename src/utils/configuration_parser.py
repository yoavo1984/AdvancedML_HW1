def parse_configuration_file(file_name):
    dict = {}
    with open(file_name) as config_file:
        lines = config_file.readlines()
        for line in lines:
            if line[0] == "#":
                continue

            key, val = line.split(":")

            # Remove white spaces and end of line.
            key = key.replace(" ", "").replace("\n", "")
            val = val.replace(" ", "").replace("\n", "")

            dict[key] = val

    process_dict(dict)
    return dict

def process_dict(dict):
    # turn lambda into array.
    lambdas = dict['lambda'].split(",")
    dict['lambda'] = list(map(float,lambdas))
    dict['d'] = int(dict['d'])

    if dict['algorithm'] == 'sgd':
        dict['alpha'] = float(dict['alpha'])
        dict['epochs'] = int(dict['epochs'])


    if dict['algorithm'] == 'als':
        dict['epsillon'] = float(dict['epsillon'])


if __name__ == "__main__":
    parse_configuration_file("../configuration")