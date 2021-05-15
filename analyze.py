def analyze(data):
    """
    Let's take a deeper look into the data
    """
    persons = data.keys()
    POIs = filter(lambda person: data[person]['poi'], data)

    print('List of All People in the Data: ')
    print(', '.join(persons))
    print('')
    print('Total Number of People:', len(persons))
    print('')
    print('Number of POIs in the Dataset:', len(POIs))
    print('')
    print('List of each individual POIs:', ', '.join(POIs))
    print('')


def fix(data):
    """
    Fixes data points by performing replacements
    """
    # Replace NaN values for zeros
    ff = [
        'salary',
        'deferral_payments',
        'total_payments',
        'loan_advances',
        'bonus',
        'restricted_stock_deferred',
        'deferred_income',
        'total_stock_value',
        'expenses',
        'exercised_stock_options',
        'other',
        'long_term_incentive',
        'restricted_stock',
        'director_fees',
        'from_poi_to_this_person',
        'from_this_person_to_poi',
        'to_messages',
        'from_messages'
    ]

    for f in ff:
        for person in data:
            if data[person][f] == 'NaN':
                data[person][f] = 0

    return data