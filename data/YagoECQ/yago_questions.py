yago_topic_to_qfs = {
    "http://schema.org/about": {
        "closed": [
            "Q: Is '{entity}' about {answer}?\nA:",
            "Q: Does {answer} pertain to '{entity}'?\nA:",
        ],
        "open": ["Q: What is '{entity}' about?\nA:", "'{entity}' is about"],
    },
    "http://schema.org/actor": {
        "closed": [
            "Q: Is {answer} an actor or actress in '{entity}'?\nA:",
            "Q: Does '{entity}' feature {answer} as an actor or actress?\nA:",
        ],
        "open": [
            "Q: Who acts in the movie, show, or play '{entity}'?\nA:",
            "'{entity}' features the actor or actress",
        ],
    },
    "reverse-http://schema.org/actor": {
        "closed": [
            "Q: Is {entity} an actor or actress in '{answer}'?\nA:",
            "Q: Does '{answer}' feature {entity} as an actor or actress?\nA:",
        ],
        "open": [
            "Q: What is a movie, show, or play that {entity} acts in?\nA:",
            "{entity} acts in",
        ],
    },
    "http://schema.org/address": {
        "closed": [
            "Q: Is {entity} located at the address '{answer}'?\nA:",
            "Q: Is '{answer}' the address of {entity}?\nA:",
        ],
        "open": ["Q: What is the address of {entity}?\nA:", "{entity} is located at"],
    },
    "http://schema.org/administrates": {
        "closed": [
            "Q: Does {entity} administrate {answer}?\nA:",
            "Q: Is {answer} administrated by {entity}?\nA:",
        ],
        "open": ["Q: What does {entity} administrate?\nA:", "{entity} administrates"],
    },
    "http://schema.org/affiliation": {
        "closed": [
            "Q: Is {entity} affiliated with {answer}?\nA:",
            "Q: Does {answer} have an affiliation with {entity}?\nA:",
        ],
        "open": [
            "Q: What organization or institution is affiliated with {entity}?\nA:",
            "{entity} is affiliated with",
        ],
    },
    "http://schema.org/alternateName": {
        "closed": [
            "Q: Is {entity} also known as {answer}?\nA:",
            "Q: Is {answer} an alternate name for {entity}?\nA:",
        ],
        "open": [
            "Q: What is an alternate name associated with {entity}?\nA:",
            "An alternate name of {entity} is",
        ],
    },
    "http://schema.org/alumniOf": {
        "closed": [
            "Q: Is {entity} an alum of {answer}?\nA:",
            "Q: Does {answer} count {entity} as one of its alumni?\nA:",
        ],
        "open": [
            "Q: Which institution or organization is {entity} an alum of?\nA:",
            "{entity} is an alum of",
        ],
    },
    # "http://schema.org/area": {
    #     "closed": [
    #         "Q: Is {entity} about {answer}?\nA:",
    #         "Q: Does {answer} pertain to {entity}?\nA:",
    #     ],
    #     "open": ["Q: What is {entity} about?\nA:", "{entity} is about"],
    # },
    "http://schema.org/author": {
        "closed": [
            "Q: Is {answer} an author of '{entity}'?\nA:",
            "Q: Was '{entity}' authored by {answer}?\nA:",
        ],
        "open": ["Q: Who authored {entity}?\nA:", "{entity} was authored by"],
    },
    "reverse-http://schema.org/author": {
        "closed": [
            "Q: Is {entity} an author of '{answer}'?\nA:",
            "Q: Was '{answer}' authored by {entity}?\nA:",
        ],
        "open": ["Q: What is a work authored by {entity}?\nA:", "{entity} is an author of"],
    },
    "http://schema.org/award": {
        "closed": [
            "Q: Was {entity} ever awarded the {answer}?\nA:",
            "Q: Has the {answer} ever been awarded to {entity}?\nA:",
        ],
        "open": [
            "Q: What is an award received by {entity}?\nA:",
            "{entity} was awarded the",
        ],
    },
    "http://schema.org/birthDate": {
        "closed": [
            "Q: Is {entity}'s birth date {answer}?\nA:",
            "Q: Is {answer} the birth date of {entity}?\nA:",
        ],
        "open": ["Q: When was {entity} born?\nA:", "{entity}'s birth date is"],
    },
    "http://schema.org/birthPlace": {
        "closed": [
            "Q: Was {entity} born in {answer}?\nA:",
            "Q: Is {answer} the birth place of {entity}?\nA:",
        ],
        "open": ["Q: Where was {entity} born?\nA:", "{entity} was born in"],
    },
    "http://schema.org/children": {
        "closed": [
            "Q: Is {entity} a parent of {answer}?\nA:",
            "Q: Is {answer} a child of {entity}?\nA:",
        ],
        "open": [
            "Q: Who is a child of {entity}?\nA:",
            "{entity} has children including",
        ],
    },
    "http://schema.org/contentLocation": {
        "closed": [
            "Q: Does '{entity}' take place in {answer}?\nA:",
            "Q: IS {answer} the setting of '{entity}'?\nA:",
        ],
        "open": [
            "Q: Where does '{entity}' take place?\nA:",
            "'{entity}' takes place in",
        ],
    },
    "http://schema.org/dateCreated": {
        "closed": [
            "Q: Was '{entity}' created on {answer}?\nA:",
            "Q: Is {answer} the creation date of '{entity}'?\nA:",
        ],
        "open": ["Q: When was '{entity}' created?\nA:", "'{entity}' was created on"],
    },
    "http://schema.org/deathDate": {
        "closed": [
            "Q: Did {entity} die on {answer}?\nA:",
            "Q: Is {answer} the death date of {entity}?\nA:",
        ],
        "open": ["Q: When did {entity} die?\nA:", "{entity} died on"],
    },
    "http://schema.org/deathPlace": {
        "closed": [
            "Q: Did {entity} die in {answer}?\nA:",
            "Q: Is {answer} the place of death for {entity}?\nA:",
        ],
        "open": ["Q: Where did {entity} die?\nA:", "{entity} died in"],
    },
    "http://schema.org/demonym": {
        "closed": [
            "Q: Are the inhabitants of {entity} called {answer}?\nA:",
            "Q: Is {answer} the name of inhabitants of {entity}?\nA:",
        ],
        "open": ["Q: What are the inhabitants of {entity} called?\nA:", "People from {entity} are known as"],
    },
    "http://schema.org/director": {
        "closed": [
            "Q: Is {answer} the director of '{entity}'?\nA:",
            "Q: Is '{entity}' directed by {answer}?\nA:",
        ],
        "open": ["Q: Who directed the work '{entity}'?\nA:", "'{entity}' is directed by"],
    },
    "reverse-http://schema.org/director": {
        "closed": [
            "Q: Is {entity} the director of '{answer}'?\nA:",
            "Q: Is '{answer}' directed by {entity}?\nA:",
        ],
        "open": ["Q: What is a work directed by {entity}?\nA:", "{entity} is the director of"],
    },
    "http://schema.org/dissolutionDate": {
        "closed": [
            "Q: Did {entity} dissolve on {answer}?\nA:",
            "Q: Is {answer} the dissolution date of {entity}?\nA:",
        ],
        "open": ["Q: When was the dissolution date of {entity}?\nA:", "{entity} dissolved on"],
    },
    "http://schema.org/duns": {
        "closed": [
            "Q: Is {entity}'s DUNS number {answer}?\nA:",
            "Q: Does {answer} represent {entity}'s DUNS number?\nA:",
        ],
        "open": [
            "Q: What is {entity}'s DUNS number?\nA:",
            "{entity}'s DUNS number is",
        ],
    },
    "http://schema.org/duration": {
        "closed": [
            "Q: Is the duration of '{entity}' {answer} minutes long?\nA:",
            "Q: Is {answer} the length of the duration of '{entity}'?\nA:",
        ],
        "open": [
            "Q: How long is the duration of '{entity}'?\nA:",
            "The duration of '{entity}' is",
        ],
    },
    "http://schema.org/editor": {
        "closed": [
            "Q: Is {entity} an editor of '{answer}'/?\nA:",
            "Q: Was '{answer}' edited by {entity}?\nA:",
        ],
        "open": ["Q: What is {entity} an editor of?\nA:", "{entity} is an editor for"],
    },
    "reverse-http://schema.org/editor": {
        "closed": [
            "Q: Is {answer} an editor of '{entity}'/?\nA:",
            "Q: Was '{entity}' edited by {answer}?\nA:",
        ],
        "open": ["Q: Who was an editor of '{entity}'?\nA:", "'{entity}' was edited by"],
    },
    "http://schema.org/elevation": {
        "closed": [
            "Q: Is the elevation of {entity} {answer}?\nA:",
            "Q: Does {answer} represent the elevation of {entity}?\nA:",
        ],
        "open": ["Q: What is the elevation of {entity}?\nA:", "The elevation of {entity} is"],
    },
    "http://schema.org/endDate": {
        "closed": [
            "Q: Did {entity} end on {answer}?\nA:",
            "Q: Is {answer} the end date of {entity}?\nA:",
        ],
        "open": ["Q: When did {entity} end?\nA:", "{entity} ended on"],
    },
    "http://schema.org/founder": {
        "closed": [
            "Q: Is {answer} a founder of '{entity}'?\nA:",
            "Q: Was '{entity}' founded by {answer}?\nA:",
        ],
        "open": ["Q: Who founded '{entity}'?\nA:", "'{entity}' was founded by"],
    },
    "reverse-http://schema.org/founder": {
        "closed": [
            "Q: Is {entity} a founder of '{answer}'?\nA:",
            "Q: Was '{answer}' founded by {entity}?\nA:",
        ],
        "open": ["Q: What is {entity} a founder of?\nA:", "{entity} is a founder of"],
    },
    "http://schema.org/gender": {
        "closed": [
            "Q: Is the gender of {entity} {answer}?\nA:",
            "Q: Does {answer} represent {entity}'s gender?\nA:",
        ],
        "open": ["Q: What is the gender of {entity}?\nA:", "The gender of {entity} is"],
    },
    # "http://schema.org/geo": {
    #     "closed": [
    #         "Q: Is {entity} related to {answer} by geo?\nA:",
    #         "Q: Does {answer} point to {entity} as its geo location?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is the geo location of {entity}?\nA:",
    #         "{entity} is the geo location for",
    #     ],
    # },
    "http://schema.org/gtin": {
        "closed": [
            "Q: Is the GTIN of '{entity}' {answer}?\nA:",
            "Q: Is {answer} the GTIN of '{entity}'?\nA:",
        ],
        "open": ["Q: What is the GTIN of '{entity}'?\nA:", "The GTIN of '{entity}' is"],
    },
    "http://schema.org/highestPoint": {
        "closed": [
            "Q: Is {entity}'s highest point {answer}?\nA:",
            "Q: Is {answer} the highest point of {entity}?\nA:",
        ],
        "open": [
            "Q: What is the highest point of {entity}?\nA:",
            "The highest point of {entity} is",
        ],
    },
    "http://schema.org/homeLocation": {
        "closed": [
            "Q: Did {answer} ever live in {entity}?\nA:",
            "Q: Was {entity} ever a place in which {answer} lived?\nA:",
        ],
        "open": [
            "Q: Who lived in {entity}?\nA:",
            "Someone who lived in {entity} is",
        ],
    },
    "reverse-http://schema.org/homeLocation": {
        "closed": [
            "Q: Did {entity} ever live in {answer}?\nA:",
            "Q: Was {answer} ever a place in which {entity} lived?\nA:",
        ],
        "open": [
            "Q: Where has {entity} lived?\nA:",
            "{entity} lived in",
        ],
    },
    "http://schema.org/humanDevelopmentIndex": {
        "closed": [
            "Q: Is {entity}'s Human Development Index {answer}?\nA:",
            "Q: Is {answer} the  Human Development Index of {entity}?\nA:",
        ],
        "open": [
            "Q: What is {entity}'s Human Development Index?\nA:",
            "{entity}'s Human Development Index is",
        ],
    },
    "http://schema.org/iataCode": {
        "closed": [
            "Q: Is {entity}'s IATA code {answer}?\nA:",
            "Q: Is {answer} the IATA code of {entity}?\nA:",
        ],
        "open": ["Q: What is the IATA code of {entity}?\nA:", "The IATA code of {entity} is"],
    },
    "http://schema.org/icaoCode": {
        "closed": [
            "Q: Is {entity}'s ICAO code {answer}?\nA:",
            "Q: Is {answer} the ICAO code of {entity}?\nA:",
        ],
        "open": ["Q: What is the ICAO code of {entity}?\nA:", "The ICAO code of {entity} is"],
    },
    "http://schema.org/illustrator": {
        "closed": [
            "Q: Is '{entity}' illustrated by {answer}?\nA:",
            "Q: Is {answer} an illustrator for '{entity}'?\nA:",
        ],
        "open": [
            "Q: Who is an illustrator of '{entity}'?\nA:",
            "'{entity}' is illustrated by",
        ],
    },
    "reverse-http://schema.org/illustrator": {
        "closed": [
            "Q: Is '{answer}' illustrated by {entity}?\nA:",
            "Q: Is {entity} an illustrator for '{answer}'?\nA:",
        ],
        "open": [
            "Q: What is one work illustrated by {entity}?\nA:",
            "{entity} illustrated",
        ],
    },
    "http://schema.org/inLanguage": {
        "closed": [
            "Q: Is '{entity}' in the {answer} language?\nA:",
            "Q: Is {answer} the language used in '{entity}'?\nA:",
        ],
        "open": [
            "Q: What language is '{entity}' in?\nA:",
            "'{entity}' is in the language",
        ],
    },
    "http://schema.org/influencedBy": {
        "closed": [
            "Q: Is {entity} influenced by {answer}?\nA:",
            "Q: Does {answer} influence {entity}?\nA:",
        ],
        "open": [
            "Q: Who influenced {entity}?\nA:",
            "{entity} was influenced by",
        ],
    },
    "http://schema.org/isbn": {
        "closed": [
            "Q: Is the ISBN of '{entity}' {answer}?\nA:",
            "Q: Is {answer} the ISBN of '{entity}'?\nA:",
        ],
        "open": ["Q: What is the ISBN of '{entity}'?\nA:", "The ISBN of '{entity}' is"],
    },
    "http://schema.org/iswcCode": {
        "closed": [
            "Q: Is the ISWC of '{entity}' {answer}?\nA:",
            "Q: Is {answer} the ISWC of '{entity}'?\nA:",
        ],
        "open": ["Q: What is the ISWC of '{entity}'?\nA:", "The ISWC of '{entity}' is"],
    },
    "http://schema.org/knowsLanguage": {
        "closed": [
            "Q: Does {entity} know the {answer} language?\nA:",
            "Q: Is {answer} a language known by {entity}?\nA:",
        ],
        "open": [
            "Q: What language does {entity} know?\nA:",
            "{entity} knows the language",
        ],
    },
    "http://schema.org/leader": {
        "closed": [
            "Q: Is {answer} the leader of {entity}?\nA:",
            "Q: Is {entity} led by {answer}?\nA:",
        ],
        "open": ["Q: Who is the leader of {entity}?\nA:", "The leader of {entity} is"],
    },
    "reverse-http://schema.org/leader": {
        "closed": [
            "Q: Is {entity} the leader of {answer}?\nA:",
            "Q: Is {answer} led by {entity}?\nA:",
        ],
        "open": ["Q: What is {entity} the leader of?\nA:", "{entity} is the leader of"],
    },
    "http://schema.org/leiCode": {
        "closed": [
            "Q: Is the LEI code of '{entity}' {answer}?\nA:",
            "Q: Is {answer} the LEI code of '{entity}'?\nA:",
        ],
        "open": ["Q: What is the LEI code of '{entity}'?\nA:", "The LEI code of '{entity}' is"],
    },
    "http://schema.org/location": {
        "closed": [
            "Q: Is {entity} located in {answer}?\nA:",
            "Q: Is {answer} the location of {entity}?\nA:",
        ],
        "open": [
            "Q: Where is {entity} located?\nA:",
            "{entity} is located in",
        ],
    },
    "http://schema.org/locationCreated": {
        "closed": [
            "Q: Was {entity} created in {answer}?\nA:",
            "Q: Is {answer} the creation place of {entity}?\nA:",
        ],
        "open": ["Q: Where was {entity} created?\nA:", "{entity} was created in"],
    },
    "http://schema.org/lowestPoint": {
        "closed": [
            "Q: Is {entity}'s lowest point {answer}?\nA:",
            "Q: Is {answer} the lowest point of {entity}?\nA:",
        ],
        "open": [
            "Q: What is the lowest point of {entity}?\nA:",
            "The lowest point of {entity} is",
        ],
    },
    "http://schema.org/lyricist": {
        "closed": [
            "Q: Is {answer} a lyricist for '{entity}'?\nA:",
            "Q: Does '{entity}' credit {answer} as a lyricist?\nA:",
        ],
        "open": [
            "Q: Who is a lyricist for '{entity}'?\nA:",
            "A lyricist for '{entity}' is a",
        ],
    },
    "reverse-http://schema.org/lyricist": {
        "closed": [
            "Q: Is {entity} a lyricist for '{answer}'?\nA:",
            "Q: Does '{answer}' credit {entity} as a lyricist?\nA:",
        ],
        "open": [
            "Q: What is a song for which {entity} is a lyricist?\nA:",
            "{entity} is a lyricist for",
        ],
    },
    # "http://schema.org/mainEntityOfPage": {
    #     "closed": [
    #         "Q: Is {entity} the main entity of the page {answer}?\nA:",
    #         "Q: Does {answer} point to {entity} as its main entity?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is the main entity of {answer}?\nA:",
    #         "{entity} is the main entity of the page",
    #     ],
    # },
    "http://schema.org/manufacturer": {
        "closed": [
            "Q: Is/was {answer} the manufacturer of {entity}?\nA:",
            "Q: Is/was {entity} manufactured by {answer}?\nA:",
        ],
        "open": [
            "Q: Who is/was the manufacturer of {entity}?\nA:",
            "{entity} is/was manufactured by",
        ],
    },
    "reverse-http://schema.org/manufacturer": {
        "closed": [
            "Q: Is/was {entity} the manufacturer of {answer}?\nA:",
            "Q: Is/was {answer} manufactured by {entity}?\nA:",
        ],
        "open": [
            "Q: What is one thing manufactured by {entity}?\nA:",
            "{entity} manufactured",
        ],
    },
    "http://schema.org/material": {
        "closed": [
            "Q: Is {entity} made of {answer}?\nA:",
            "Q: Does {answer} represent the material of {entity}?\nA:",
        ],
        "open": ["Q: What material is {entity} made of?\nA:", "{entity} is made of"],
    },
    "http://schema.org/memberOf": {
        "closed": [
            "Q: Is {entity} a member of {answer}?\nA:",
            "Q: Does {answer} count {entity} as one of its members?\nA:",
        ],
        "open": [
            "Q: What group or organization is {entity} a member of?\nA:",
            "{entity} is a member of",
        ],
    },
    "reverse-http://schema.org/memberOf": {
        "closed": [
            "Q: Is {answer} a member of {entity}?\nA:",
            "Q: Does {entity} count {answer} as one of its members?\nA:",
        ],
        "open": [
            "Q: Who is a member of {entity}?\nA:",
            "One member of {entity} is",
        ],
    },
    "http://schema.org/motto": {
        "closed": [
            "Q: Is {entity}'s motto '{answer}'?\nA:",
            "Q: Is '{answer}' the motto of {entity}?\nA:",
        ],
        "open": ["Q: What is {entity}'s motto?\nA:", "{entity}'s motto is"],
    },  # STOP HERE
    # "http://schema.org/musicBy": {
    #     "closed": [
    #         "Q: Is {entity} attributed to {answer} as its music creator?\nA:",
    #         "Q: Does {answer} credit {entity} as the music creator?\nA:",
    #     ],
    #     "open": [
    #         "Q: Who created the music for {entity}?\nA:",
    #         "The music for {entity} is by",
    #     ],
    # },
    # "http://schema.org/nationality": {
    #     "closed": [
    #         "Q: Is {entity}'s nationality {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s nationality?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is {entity}'s nationality?\nA:",
    #         "{entity}'s nationality is",
    #     ],
    # },
    # "http://schema.org/neighbors": {
    #     "closed": [
    #         "Q: Are {entity} and {answer} neighbors?\nA:",
    #         "Q: Does {answer} list {entity} as one of its neighbors?\nA:",
    #     ],
    #     "open": [
    #         "Q: Who are the neighbors of {entity}?\nA:",
    #         "{entity} has neighbors including",
    #     ],
    # },
    # "http://schema.org/numberOfEmployees": {
    #     "closed": [
    #         "Q: Does {entity} have {answer} employees?\nA:",
    #         "Q: Is {answer} the number of employees for {entity}?\nA:",
    #     ],
    #     "open": [
    #         "Q: How many employees does {entity} have?\nA:",
    #         "{entity} has a total of",
    #     ],
    # },
    # "http://schema.org/numberOfEpisodes": {
    #     "closed": [
    #         "Q: Does {entity} have {answer} episodes?\nA:",
    #         "Q: Is {answer} the number of episodes for {entity}?\nA:",
    #     ],
    #     "open": [
    #         "Q: How many episodes does {entity} have?\nA:",
    #         "{entity} has a total of",
    #     ],
    # },
    # "http://schema.org/numberOfPages": {
    #     "closed": [
    #         "Q: Does {entity} have {answer} pages?\nA:",
    #         "Q: Is {answer} the number of pages for {entity}?\nA:",
    #     ],
    #     "open": [
    #         "Q: How many pages does {entity} have?\nA:",
    #         "{entity} has a total of",
    #     ],
    # },
    # "http://schema.org/numberOfSeasons": {
    #     "closed": [
    #         "Q: Does {entity} have {answer} seasons?\nA:",
    #         "Q: Is {answer} the number of seasons for {entity}?\nA:",
    #     ],
    #     "open": [
    #         "Q: How many seasons does {entity} have?\nA:",
    #         "{entity} has a total of",
    #     ],
    # },
    # "http://schema.org/officialLanguage": {
    #     "closed": [
    #         "Q: Is {entity}'s official language {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s official language?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is {entity}'s official language?\nA:",
    #         "{entity}'s official language is",
    #     ],
    # },
    # "http://schema.org/organizer": {
    #     "closed": [
    #         "Q: Is {entity} organized by {answer}?\nA:",
    #         "Q: Does {answer} serve as the organizer for {entity}?\nA:",
    #     ],
    #     "open": ["Q: Who organizes {entity}?\nA:", "{entity} is organized by"],
    # },
    # "http://schema.org/ownedBy": {
    #     "closed": [
    #         "Q: Is {entity} owned by {answer}?\nA:",
    #         "Q: Does {answer} own {entity}?\nA:",
    #     ],
    #     "open": ["Q: Who owns {entity}?\nA:", "{entity} is owned by"],
    # },
    # "http://schema.org/owns": {
    #     "closed": [
    #         "Q: Does {entity} own {answer}?\nA:",
    #         "Q: Is {answer} owned by {entity}?\nA:",
    #     ],
    #     "open": ["Q: What does {entity} own?\nA:", "{entity} owns"],
    # },
    # "http://schema.org/parentTaxon": {
    #     "closed": [
    #         "Q: Is {entity} a parent taxon of {answer}?\nA:",
    #         "Q: Does {answer} have {entity} as a parent taxon?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is the parent taxon of {answer}?\nA:",
    #         "{entity} is a parent taxon of",
    #     ],
    # },
    # "http://schema.org/performer": {
    #     "closed": [
    #         "Q: Is {entity} a performer for {answer}?\nA:",
    #         "Q: Does {answer} list {entity} as a performer?\nA:",
    #     ],
    #     "open": ["Q: Who performs for {answer}?\nA:", "{entity} is a performer for"],
    # },
    # "http://schema.org/populationNumber": {
    #     "closed": [
    #         "Q: Is {entity}'s population {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s population?\nA:",
    #     ],
    #     "open": ["Q: What is {entity}'s population?\nA:", "{entity}'s population is"],
    # },
    # "http://schema.org/postalCode": {
    #     "closed": [
    #         "Q: Is {entity}'s postal code {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s postal code?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is the postal code of {entity}?",
    #         "The postal code of {entity} is",
    #     ],
    # },
    # "http://schema.org/recordLabel": {
    #     "closed": [
    #         "Q: Is {entity} signed to {answer} as a record label?\nA:",
    #         "Q: Does {answer} serve as {entity}'s record label?\nA:",
    #     ],
    #     "open": [
    #         "Q: Who is {entity}'s record label?\nA:",
    #         "{entity}'s record label is",
    #     ],
    # },
    # "http://schema.org/sameAs": {
    #     "closed": [
    #         "Q: Is {entity} the same as {answer}?\nA:",
    #         "Q: Does {answer} represent the same entity as {entity}?\nA:",
    #     ],
    #     "open": [
    #         "Q: What entity is {entity} the same as?\nA:",
    #         "{entity} is the same as",
    #     ],
    # },
    # "http://schema.org/sponsor": {
    #     "closed": [
    #         "Q: Is {entity} sponsored by {answer}?\nA:",
    #         "Q: Does {answer} serve as a sponsor for {entity}?\nA:",
    #     ],
    #     "open": ["Q: Who sponsors {entity}?\nA:", "{entity} is sponsored by"],
    # },
    # "http://schema.org/spouse": {
    #     "closed": [
    #         "Q: Is {entity} married to {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s spouse?\nA:",
    #     ],
    #     "open": ["Q: Who is {entity}'s spouse?\nA:", "{entity}'s spouse is"],
    # },
    # "http://schema.org/startDate": {
    #     "closed": [
    #         "Q: Does {entity} start on {answer}?\nA:",
    #         "Q: Is {answer} {entity}'s start date?\nA:",
    #     ],
    #     "open": ["Q: When does {entity} start?\nA:", "{entity} starts on"],
    # },
    # "http://schema.org/superEvent": {
    #     "closed": [
    #         "Q: Is {entity} a super event of {answer}?\nA:",
    #         "Q: Does {answer} have {entity} as its super event?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is the super event of {answer}?\nA:",
    #         "{entity} is a super event of",
    #     ],
    # },
    # "http://schema.org/unemploymentRate": {
    #     "closed": [
    #         "Q: Is {entity}'s unemployment rate {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s unemployment rate?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is {entity}'s unemployment rate?\nA:",
    #         "{entity}'s unemployment rate is",
    #     ],
    # },
    # "http://schema.org/url": {
    #     "closed": [
    #         "Q: Is {entity}'s URL {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s URL?\nA:",
    #     ],
    #     "open": ["Q: What is {entity}'s URL?\nA:", "{entity}'s URL is"],
    # },
    # "http://schema.org/worksFor": {
    #     "closed": [
    #         "Q: Does {entity} work for {answer}?\nA:",
    #         "Q: Is {answer} {entity}'s employer?\nA:",
    #     ],
    #     "open": ["Q: Who does {entity} work for?\nA:", "{entity} works for"],
    # },
    # "http://yago-knowledge.org/resource/academicDegree": {
    #     "closed": [
    #         "Q: Is {entity} awarded an academic degree in {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s academic degree?\nA:",
    #     ],
    #     "open": [
    #         "Q: What academic degree does {entity} hold?\nA:",
    #         "{entity} holds an academic degree in",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/appearsIn": {
    #     "closed": [
    #         "Q: Does {entity} appear in {answer}?\nA:",
    #         "Q: Is {answer} a place where {entity} appears?\nA:",
    #     ],
    #     "open": ["Q: Where does {entity} appear?\nA:", "{entity} appears in"],
    # },
    # "http://yago-knowledge.org/resource/beliefSystem": {
    #     "closed": [
    #         "Q: Does {entity} believe in {answer}?\nA:",
    #         "Q: Is {answer} {entity}'s belief system?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is {entity}'s belief system?\nA:",
    #         "{entity}'s belief system is",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/candidateIn": {
    #     "closed": [
    #         "Q: Is {entity} a candidate in {answer}?\nA:",
    #         "Q: Does {answer} involve {entity} as a candidate?\nA:",
    #     ],
    #     "open": [
    #         "Q: For what is {entity} a candidate?\nA:",
    #         "{entity} is a candidate in",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/capital": {
    #     "closed": [
    #         "Q: Is {entity}'s capital {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s capital?\nA:",
    #     ],
    #     "open": ["Q: What is {entity}'s capital?\nA:", "{entity}'s capital is"],
    # },
    # "http://yago-knowledge.org/resource/conferredBy": {
    #     "closed": [
    #         "Q: Is {entity} conferred by {answer}?\nA:",
    #         "Q: Does {answer} confer {entity}?\nA:",
    #     ],
    #     "open": ["Q: Who confers {entity}?\nA:", "{entity} is conferred by"],
    # },
    # "http://yago-knowledge.org/resource/consumes": {
    #     "closed": [
    #         "Q: Does {entity} consume {answer}?\nA:",
    #         "Q: Is {answer} something {entity} consumes?\nA:",
    #     ],
    #     "open": ["Q: What does {entity} consume?\nA:", "{entity} consumes"],
    # },
    # "http://yago-knowledge.org/resource/director": {
    #     "closed": [
    #         "Q: Is {entity} a director in {answer}?\nA:",
    #         "Q: Does {answer} have {entity} as a director?\nA:",
    #     ],
    #     "open": ["Q: Who directs {answer}?\nA:", "{entity} is a director in"],
    # },
    # "http://yago-knowledge.org/resource/distanceFromEarth": {
    #     "closed": [
    #         "Q: Is {entity}'s distance from Earth {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s distance from Earth?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is {entity}'s distance from Earth?\nA:",
    #         "{entity}'s distance from Earth is",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/doctoralAdvisor": {
    #     "closed": [
    #         "Q: Is {entity} advised by {answer} for doctoral studies?\nA:",
    #         "Q: Does {answer} serve as {entity}'s doctoral advisor?\nA:",
    #     ],
    #     "open": [
    #         "Q: Who is {entity}'s doctoral advisor?\nA:",
    #         "{entity}'s doctoral advisor is",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/flowsInto": {
    #     "closed": [
    #         "Q: Does {entity} flow into {answer}?\nA:",
    #         "Q: Is {answer} where {entity} flows into?\nA:",
    #     ],
    #     "open": ["Q: Where does {entity} flow?\nA:", "{entity} flows into"],
    # },
    # "http://yago-knowledge.org/resource/follows": {
    #     "closed": [
    #         "Q: Does {entity} follow {answer}?\nA:",
    #         "Q: Is {answer} what {entity} follows?\nA:",
    #     ],
    #     "open": ["Q: Who or what does {entity} follow?\nA:", "{entity} follows"],
    # },
    # "http://yago-knowledge.org/resource/length": {
    #     "closed": [
    #         "Q: Is {entity}'s length {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s length?\nA:",
    #     ],
    #     "open": ["Q: What is {entity}'s length?\nA:", "{entity}'s length is"],
    # },
    # "http://yago-knowledge.org/resource/luminosity": {
    #     "closed": [
    #         "Q: Is {entity}'s luminosity {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s luminosity?\nA:",
    #     ],
    #     "open": ["Q: What is {entity}'s luminosity?\nA:", "{entity}'s luminosity is"],
    # },
    # "http://yago-knowledge.org/resource/mass": {
    #     "closed": [
    #         "Q: Is {entity}'s mass {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s mass?\nA:",
    #     ],
    #     "open": ["Q: What is {entity}'s mass?\nA:", "{entity}'s mass is"],
    # },
    # "http://yago-knowledge.org/resource/notableWork": {
    #     "closed": [
    #         "Q: Is {entity} associated with {answer} as a notable work?\nA:",
    #         "Q: Does {answer} represent {entity}'s notable work?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is {entity}'s notable work?\nA:",
    #         "{entity}'s notable work is",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/parallax": {
    #     "closed": [
    #         "Q: Is {entity}'s parallax {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s parallax?\nA:",
    #     ],
    #     "open": ["Q: What is {entity}'s parallax?\nA:", "{entity}'s parallax is"],
    # },
    # "http://yago-knowledge.org/resource/parentBody": {
    #     "closed": [
    #         "Q: Is {entity} a parent body of {answer}?\nA:",
    #         "Q: Does {answer} have {entity} as a parent body?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is {entity}'s parent body?\nA:",
    #         "{entity} is a parent body of",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/participant": {
    #     "closed": [
    #         "Q: Is {entity} a participant in {answer}?\nA:",
    #         "Q: Does {answer} involve {entity} as a participant?\nA:",
    #     ],
    #     "open": [
    #         "Q: Who participates in {answer}?\nA:",
    #         "{entity} is a participant in",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/playsIn": {
    #     "closed": [
    #         "Q: Does {entity} play in {answer}?\nA:",
    #         "Q: Is {answer} where {entity} plays?\nA:",
    #     ],
    #     "open": ["Q: Where does {entity} play?\nA:", "{entity} plays in"],
    # },
    # "http://yago-knowledge.org/resource/radialVelocity": {
    #     "closed": [
    #         "Q: Is {entity}'s radial velocity {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s radial velocity?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is {entity}'s radial velocity?\nA:",
    #         "{entity}'s radial velocity is",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/replaces": {
    #     "closed": [
    #         "Q: Does {entity} replace {answer}?\nA:",
    #         "Q: Is {answer} replaced by {entity}?\nA:",
    #     ],
    #     "open": ["Q: What does {entity} replace?\nA:", "{entity} replaces"],
    # },
    # "http://yago-knowledge.org/resource/sportNumber": {
    #     "closed": [
    #         "Q: Is {entity}'s sport number {answer}?\nA:",
    #         "Q: Does {answer} represent {entity}'s sport number?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is {entity}'s sport number?\nA:",
    #         "{entity}'s sport number is",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/studentOf": {
    #     "closed": [
    #         "Q: Is {entity} a student of {answer}?\nA:",
    #         "Q: Does {answer} teach {entity}?\nA:",
    #     ],
    #     "open": ["Q: Who is {entity}'s teacher?\nA:", "{entity} is a student of"],
    # },
    # "http://yago-knowledge.org/resource/studentsCount": {
    #     "closed": [
    #         "Q: Does {entity} have {answer} students?\nA:",
    #         "Q: Is {answer} the count of {entity}'s students?\nA:",
    #     ],
    #     "open": [
    #         "Q: How many students does {entity} have?\nA:",
    #         "{entity} has a total of",
    #     ],
    # },
    # "http://yago-knowledge.org/resource/terminus": {
    #     "closed": [
    #         "Q: Is {entity} the terminus of {answer}?\nA:",
    #         "Q: Does {answer} have {entity} as its terminus?\nA:",
    #     ],
    #     "open": [
    #         "Q: What is the terminus of {answer}?\nA:",
    #         "{entity} is the terminus of",
    #     ],
    # },
}
