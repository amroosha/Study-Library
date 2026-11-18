t = int(input())

while t > 0:
    t -= 1
    notes = int(input())

    v = [int(x) for x in input().split()]

    diff = []
    ans = 0

    for i in range(notes - 4):
        diff.append(v[i] + v[i + 2] - v[i + 4])

    frequencies = {}
    for i, current in enumerate(diff):
        matching_previous = frequencies.get(current, 0)


        if i >= 2 and diff[i - 2] == current:
            matching_previous -= 1
        if i >= 4 and diff[i - 4] == current:
            matching_previous -= 1

        ans += matching_previous
        frequencies[current] = frequencies.get(current, 0) + 1

    print(ans)