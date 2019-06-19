def read_harmonic_function(filename):
    offset = 5
    coordinates = []
    potentials = []

    with open(filename, 'r') as fin:
        for line_num, line in enumerate(fin):
            if line_num < offset:
                vals = map(int, line.split())
                coordinates.append((vals[0], vals[1]))
                potentials.append(vals[2])

    print(potentials)


def main():
    pass

if __name__ == '__main__':
    main()
