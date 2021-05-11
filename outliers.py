from plots import scatter


def find_outliers(data):

    """
    Finds the outliers in the dataset
    """

    scatter(data, ['salary', 'bonus'])

    print('Finding possible outliers...')
    possible_outliers = []
    for person in data:
        if data[person]['salary'] > 800000 or data[person]['bonus'] > 6000000 or \
                (data[person]['salary'] == 0 and data[person]['bonus'] == 0 and
                 data[person]['total_payments'] == 0 and data[person]['from_poi_to_this_person'] == 0 and
                 data[person]['total_stock_value'] == 0 and data[person]['from_this_person_to_poi'] == 0 and
                 data[person]['from_messages'] == 'NaN' and data[person]['to_messages'] == 'NaN'):
            possible_outliers.append(person)

    # There is one key which is not an actual name:
    possible_outliers.append('THE TRAVEL AGENCY IN THE PARK')

    print('Found this many outliers: {:,} "{}"'.format(len(possible_outliers), ', '.join(possible_outliers)))
    # Let's examine now the possible outliers
    outliers = []
    for possible_outlier in possible_outliers:
        if data[possible_outlier]["poi"]:
            print('  -> "{}" is excluded for being a POI'.format(possible_outlier))
        elif data[possible_outlier]['from_poi_to_this_person'] + data[possible_outlier]['from_this_person_to_poi'] > 100:
            print('  -> "{}" is excluded for having high interactions with a POI'.format(possible_outlier))
        else:
            outliers.append(possible_outlier)

    print('Found {:,} actual outliers, "{}"'.format(len(outliers), ', '.join(outliers)))

    return outliers


def remove_outliers(data, outliers):
    """
    Removes the outliers from the dataset

    :param data: Dict with the data associated to each person
    :return: same data without the outliers
    """
    ds = dict(data)
    print("Items on the DS before removal:", len(ds))

    for outlier in outliers:
        ds.pop(outlier, None)

    print('Removed {:,} outliers from the dataset'.format(len(outliers)))
    print("Items on the DS after removal:", len(ds))

    return ds
