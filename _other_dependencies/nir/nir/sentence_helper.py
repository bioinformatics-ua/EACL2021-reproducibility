# New sentencer splitter

def has_digit_close(word):
    index = word.find(",")
    _low = max(0,index-1)
    _high = min(len(word)-1, index+1)

    return word[_low].isdigit() or word[_high].isdigit()


def sentence_splitter(s):

    white_spaced_words = s.split(" ")

    MEAN = 25

    start_index = 0
    end_index = 1
    previous_candidate_index = None
    next_candidate_index = None

    # build sentences
    sentences = []
    while end_index < len(white_spaced_words):
        #print(end_index, previous_candidate_index, next_candidate_index)
        
        if "," in white_spaced_words[end_index] and not has_digit_close(white_spaced_words[end_index]):
            # possible candidate
            if previous_candidate_index is None:
                previous_candidate_index = end_index
            elif next_candidate_index is None:
                next_candidate_index = end_index

                if len(white_spaced_words[start_index:next_candidate_index+1]) < MEAN:
                    previous_candidate_index = next_candidate_index
                    next_candidate_index = None

        if previous_candidate_index is not None and next_candidate_index is not None:

            # chose the closest
            _previous = white_spaced_words[start_index:previous_candidate_index+1]
            _next = white_spaced_words[start_index:next_candidate_index+1]

            if MEAN - len(_previous) < len(_next) - MEAN:
                # previous
                sentences.append(_previous)

                start_index = previous_candidate_index + 1
                previous_candidate_index = None
                next_candidate_index = None
            else:
                # next
                sentences.append(_next)

                start_index = next_candidate_index + 1

                previous_candidate_index = None
                next_candidate_index = None

            end_index = start_index

        end_index += 1
    
    if previous_candidate_index is not None:
        #print("append last?")
        sentences.append(white_spaced_words[previous_candidate_index+1:])

    return list(map( lambda s:" ".join(s), sentences))