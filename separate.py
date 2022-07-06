import pickle
import numpy as np

def main(n_split):
    with open('./data/raw_link.pkl', 'rb') as f:
        link_list = pickle.load(f)

    database_title = np.load('./data/database_title.npy')
    database_abst = np.load('./data/database_abst.npy')
    with open('./data/raw_title.pkl', 'rb') as f:
        raw_title = pickle.load(f)
    with open('./data/raw_abst.pkl', 'rb') as f:
        raw_abst = pickle.load(f)

    assert len(database_title) == len(database_abst)
    assert len(raw_title) == len(raw_abst)
    assert len(database_title) == len(raw_title)
    
    split_size = len(link_list) // n_split
    for i in range(n_split):
        if i < n_split - 1:
            np.save('./data/database_title/database_title{}.npy'.format(i+1), database_title[i*split_size:(i+1)*split_size])
            np.save('./data/database_abst/database_abst{}.npy'.format(i+1), database_abst[i*split_size:(i+1)*split_size])
            with open('./data/raw_title/raw_title{}.pkl'.format(i+1), 'wb') as f:
                pickle.dump(raw_title[i*split_size:(i+1)*split_size], f)
            with open('./data/raw_abst/raw_abst{}.pkl'.format(i+1), 'wb') as f:
                pickle.dump(raw_abst[i*split_size:(i+1)*split_size], f)
        else:
            np.save('./data/database_title/database_title{}.npy'.format(i+1), database_title[i*split_size:len(link_list)])
            np.save('./data/database_abst/database_abst{}.npy'.format(i+1), database_abst[i*split_size:len(link_list)])
            with open('./data/raw_title/raw_title{}.pkl'.format(i+1), 'wb') as f:
                pickle.dump(raw_title[i*split_size:len(link_list)], f)
            with open('./data/raw_abst/raw_abst{}.pkl'.format(i+1), 'wb') as f:
                pickle.dump(raw_abst[i*split_size:len(link_list)], f)
    print(database_title[i*split_size:(i+1)*split_size].shape)

if __name__ == "__main__":
    main(n_split=30)