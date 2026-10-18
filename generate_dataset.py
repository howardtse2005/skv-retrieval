"""Generate deterministic synthetic KV retrieval datasets."""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

DEFAULT_SIZES = (75, 140, 300)
DEFAULT_CONDITIONS = ("high", "low")
POSITIONS = ("beginning", "25th_percentile", "50th_percentile",
             "75th_percentile", "end")
KEY_TYPES = (
    "date of birth", "favorite color", "pet name", "phone number", "employee ID",
    "street address", "city of residence", "country of residence", "nationality",
    "language spoken", "occupation", "department", "job title", "company name",
    "school name", "university name", "degree", "major subject", "favorite electronic brand",
    "favorite food", "favorite drink", "favorite movie", "favorite book",
    "favorite song", "favorite band", "favorite sport", "favorite team",
    "favorite holiday", "favorite season", "favorite animal", "hobby",
    "musical instrument", "shirt size", "shoe size", "height", "weight",
    "eye color", "hair color", "blood type", "marital status", "number of siblings",
    "mother's name", "father's name", "birth city", "birth country",
    "email address", "website", "social handle", "passport number", "license number",
    "bank name", "account type", "currency", "tax region", "time zone",
    "preferred currency", "emergency contact", "emergency phone", "allergy",
    "dietary restriction", "medical condition", "insurance provider",
    "insurance number", "doctor name", "hospital name", "room number",
    "vehicle make", "vehicle model", "vehicle color", "vehicle type",
    "license plate", "flight number", "frequent flyer number", "hotel name",
    "room preference", "check-in date", "check-out date", "package carrier",
    "tracking number", "favorite website", "operating system", "device model",
    "screen size", "internet provider", "router model", "wifi network",
    "project name", "project code", "team name", "manager name", "office floor",
    "desk number", "building name", "meeting room", "access level",
    "security question", "preferred contact method", "customer tier",
    "subscription plan", "account status",
)
HIGH_SIMILARITY_EXCLUDED_KEYS = frozenset({
    "phone number", "employee ID", "street address", "email address",
    "social handle", "passport number", "license number", "insurance number",
    "emergency phone", "room number", "license plate", "flight number",
    "frequent flyer number", "tracking number", "project code", "desk number",
})
HIGH_SIMILARITY_KEYS = tuple(
    key for key in KEY_TYPES if key not in HIGH_SIMILARITY_EXCLUDED_KEYS
)
BASE_VALUES = {
    "date of birth": ("1 January 1950", "15 June 1975", "28 November 2000", "4 March 1925", "9 September 1940", "22 December 1965", "11 February 1985", "30 July 2010"),
    "favorite color": ("red", "purple", "green", "blue", "yellow", "orange", "pink", "brown"),
    "pet name": ("Fluffy", "Mittens", "Buddy", "Max", "Bella", "Charlie", "Luna", "Rocky"),
    "phone number": ("555-0100", "555-0101", "555-0102", "555-1003", "565-0104", "557-0105", "575-0126", "545-0147"),
    "employee ID": ("A7X3B9", "B4K8P2", "C9M1Q7", "D2L5R4", "E6N3S8", "F1J7T5", "G8H2U1", "H5V9W6"),
    "street address": ("12 Oak Street", "48 Maple Avenue", "7 Cedar Road", "23 Pine Lane", "56 Birch Boulevard", "34 Elm Drive", "89 Spruce Court", "15 Willow Way"),
    "city of residence": ("London", "Toronto", "Singapore", "New York", "Sydney", "Berlin", "Tokyo", "Paris"),
    "country of residence": ("United Kingdom", "Canada", "Republic of Singapore", "United States", "Australia", "Germany", "Japan", "France"),
    "nationality": ("British", "Canadian", "Singaporean", "American", "Australian", "German", "Japanese", "French"),
    "language spoken": ("English", "French", "Mandarin", "Spanish", "German", "Japanese", "Korean", "Italian"),
    "occupation": ("teacher", "engineer", "nurse", "artist", "scientist", "writer", "musician", "chef"),
    "department": ("finance", "marketing", "engineering", "human resources", "sales", "research and development", "customer service", "operations"),
    "job title": ("analyst", "manager", "director", "coordinator", "specialist", "consultant", "administrator", "supervisor"),
    "company name": ("Northstar Labs", "Cedar Systems", "Orion Works", "Pinecone Solutions", "MapleTech", "Silverline Industries", "Blue Horizon Corp", "Golden Gate Enterprises"),
    "school name": ("Oakridge School", "Riverside Academy", "Hillcrest School", "Lakeside Institute", "Maplewood School", "Cedar Valley Academy", "Pinehill School", "Silverlake Institute"),
    "university name": ("Oxford University", "McGill University", "Nanyang University", "Harvard University", "University of Sydney", "Technical University of Munich", "University of Tokyo", "Sorbonne University"),
    "degree": ("Bachelor of Arts", "Master of Science", "Doctor of Philosophy", "Bachelor of Engineering", "Master of Business Administration", "Doctor of Medicine", "Bachelor of Fine Arts", "Master of Public Health"),
    "major subject": ("history", "computer science", "biology", "mechanical engineering", "business administration", "medicine", "fine arts", "public health"),
    "favorite electronic brand": ("xiaomi", "apple", "samsung", "intel", "amd", "nvidia", "huawei", "lenovo"),
    "favorite food": ("pizza", "sushi", "pasta", "burger", "salad", "tacos", "curry", "steak"),
    "favorite drink": ("coffee", "tea", "lemonade", "smoothie", "soda", "water", "juice", "milkshake"),
    "favorite movie": ('Inception', 'Moonlight', 'Arrival', 'The Godfather', 'Spirited Away', 'The Matrix', 'Casablanca', 'Parasite'),
    "favorite book": ('Dune', 'Beloved', '1984', 'Pride and Prejudice', 'The Hobbit', 'To Kill a Mockingbird', 'The Great Gatsby', 'Sapiens'),
    "favorite song": ('Imagine', 'Hallelujah', 'Yesterday', 'Bohemian Rhapsody', 'Let It Be', 'Hotel California', 'What a Wonderful World', 'Shape of You'),
    "favorite band": ('The Beatles', 'Radiohead', 'Coldplay', 'Queen', 'Nirvana', 'Fleetwood Mac', 'Maroon 5', 'ABBA'),
    "favorite sport": ('football', 'tennis', 'basketball', 'cricket', 'baseball', 'swimming', 'volleyball', 'cycling'),
    "favorite team": ('Arsenal', 'Raptors', 'Lakers', 'Yankees', 'Warriors', 'All Blacks', 'Bayern Munich', 'Dodgers'),
    "favorite holiday": ('Christmas', 'Diwali', 'New Year', 'Eid al-Fitr', 'Thanksgiving', 'Hanukkah', 'Lunar New Year', 'Easter'),
    "favorite season": ('spring', 'summer', 'autumn', 'winter', 'monsoon', 'dry season', 'harvest season', 'late summer'),
    "favorite animal": ('dog', 'cat', 'elephant', 'lion', 'dolphin', 'penguin', 'horse', 'tiger'),
    "hobby": ('painting', 'gardening', 'photography', 'woodworking', 'knitting', 'cooking', 'birdwatching', 'cycling'),
    "musical instrument": ('piano', 'guitar', 'violin', 'drums', 'flute', 'saxophone', 'cello', 'trumpet'),
    "shirt size": ('small', 'medium', 'large', 'extra small', 'extra large', 'XXL', 'petite', 'tall'),
    "shoe size": ('size 8', 'size 9', 'size 10', 'size 6', 'size 7', 'size 11', 'size 12', 'size 13'),
    "height": ('170 cm', '180 cm', '190 cm', '155 cm', '165 cm', '175 cm', '185 cm', '195 cm'),
    "weight": ('60 kg', '75 kg', '90 kg', '50 kg', '65 kg', '80 kg', '100 kg', '110 kg'),
    "eye color": ('hazel', 'blue', 'emerald', 'brown', 'gray', 'amber', 'green', 'violet'),
    "hair color": ('black', 'auburn', 'blonde', 'brown', 'gray', 'red', 'silver', 'dark brown'),
    "blood type": ('A positive', 'B positive', 'O positive', 'AB positive', 'A negative', 'B negative', 'O negative', 'AB negative'),
    "marital status": ('single', 'married', 'divorced', 'engaged', 'widowed', 'separated', 'partnered', 'civil union'),
    "number of siblings": ('0 siblings', '1 sibling', '2 siblings', '3 siblings', '4 siblings', '5 siblings', '6 siblings', '7 siblings'),
    "mother's name": ('Maria', 'Susan', 'Fatima', 'Elizabeth', 'Aisha', 'Linda', 'Grace', 'Hannah'),
    "father's name": ('James', 'Robert', 'Ahmed', 'Michael', 'David', 'Daniel', 'Thomas', 'William'),
    "birth city": ('Manchester', 'Vancouver', 'Kuala Lumpur', 'Birmingham', 'Montreal', 'Penang', 'Melbourne', 'Seattle'),
    "birth country": ('England', 'Canadian territory', 'Malaysia', 'Wales', 'France', 'Thailand', 'New Zealand', 'United States'),
    "email address": ('alex@example.com', 'sam@example.com', 'lee@example.com', 'jordan@example.com', 'morgan@example.com', 'casey@example.com', 'riley@example.com', 'taylor@example.com'),
    "website": ('example.org', 'example.net', 'example.com', 'example.edu', 'example.io', 'example.co.uk', 'example.ca', 'example.au'),
    "social handle": ('@alex', '@sam', '@lee', '@jordan', '@morgan', '@casey', '@riley', '@taylor'),
    "passport number": ('P1234567', 'P2345678', 'P3456789', 'P4567890', 'P5678901', 'P6789012', 'P7890123', 'P8901234'),
    "license number": ('D123456', 'D234567', 'D345678', 'D456789', 'D567890', 'D678901', 'D789012', 'D890123'),
    "bank name": ('Barclays', 'RBC', 'DBS', 'HSBC', 'Citibank', 'Chase', 'Wells Fargo', 'Standard Chartered'),
    "account type": ('checking', 'savings', 'business', 'joint', 'student', 'trust', 'retirement', 'corporate'),
    "currency": ("GBP", "CAD", "SGD", "HKD", "EUR", "JPY", "AUD", "CHF"),
    "tax region": ('England tax zone', 'Ontario', 'Singapore tax zone', 'Scotland', 'Quebec', 'California', 'Bavaria', 'New South Wales'),
    "time zone": ('Europe/London', 'America/Toronto', 'Asia/Singapore', 'America/New_York', 'Europe/Paris', 'Asia/Tokyo', 'Australia/Sydney', 'America/Los_Angeles'),
    "preferred currency": ('pound sterling', 'Canadian dollar', 'Singapore dollar', 'US dollar', 'euro', 'Japanese yen', 'Australian dollar', 'Swiss franc'),
    "emergency contact": ('Alice Smith', 'Brian Lee', 'Carol Wong', 'David Kim', 'Emma Davis', 'Frank Martin', 'Grace Wilson', 'Henry Brown'),
    "emergency phone": ('555-0110', '555-0111', '555-0112', '555-0113', '555-0114', '555-0115', '555-0116', '555-0117'),
    "allergy": ('none', 'peanuts', 'pollen', 'shellfish', 'dairy', 'gluten', 'latex', 'dust'),
    "dietary restriction": ('no restriction', 'vegetarian', 'vegan', 'halal', 'kosher', 'gluten-free', 'pescatarian', 'low sodium'),
    "medical condition": ('no condition', 'asthma', 'migraine', 'diabetes', 'hypertension', 'eczema', 'arthritis', 'insomnia'),
    "insurance provider": ('Aviva', 'Manulife', 'AIA', 'Bupa', 'Cigna', 'UnitedHealthcare', 'Blue Cross', 'Kaiser Permanente'),
    "insurance number": ('INS10001', 'INS20002', 'INS30003', 'INS40004', 'INS50005', 'INS60006', 'INS70007', 'INS80008'),
    "doctor name": ('Dr Patel', 'Dr Chen', 'Dr Williams', 'Dr Garcia', 'Dr Singh', 'Dr Johnson', 'Dr Nguyen', 'Dr Anderson'),
    "hospital name": ('Central Hospital', 'Riverside Hospital', 'City Hospital', 'Memorial Hospital', 'St Mary Hospital', 'General Hospital', 'University Hospital', 'Mercy Hospital'),
    "room number": ('Room 101', 'Room 202', 'Room 303', 'Room 404', 'Room 505', 'Room 606', 'Room 707', 'Room 808'),
    "vehicle make": ('Toyota', 'Ford', 'Honda', 'Nissan', 'BMW', 'Mercedes-Benz', 'Hyundai', 'Subaru'),
    "vehicle model": ('Corolla', 'Focus', 'Civic', 'Altima', '3 Series', 'C-Class', 'Elantra', 'Outback'),
    "vehicle color": ('silver', 'white', 'obsidian', 'red', 'blue', 'green', 'black', 'gray'),
    "vehicle type": ('motorcycle', 'car', 'bike', 'plane', 'bus', 'train', 'scooter', 'boat'),
    "license plate": ('ABC123', 'DEF456', 'GHI789', 'JKL012', 'MNO345', 'PQR678', 'STU901', 'VWX234'),
    "flight number": ('BA101', 'AC202', 'SQ303', 'LH404', 'EK505', 'QF606', 'UA707', 'JL808'),
    "frequent flyer number": ('FF10001', 'FF20002', 'FF30003', 'FF40004', 'FF50005', 'FF60006', 'FF70007', 'FF80008'),
    "hotel name": ('Grand Hotel', 'Park Hotel', 'Harbour Hotel', 'Seaside Resort', 'City Lodge', 'Mountain Inn', 'Garden Suites', 'Royal Plaza'),
    "room preference": ('window seat', 'high floor', 'quiet room', 'king bed', 'late checkout', 'ground floor', 'city view', 'non-smoking'),
    "check-in date": ('1 April 2024', '10 May 2024', '20 June 2024', '5 July 2024', '12 August 2024', '3 September 2024', '14 October 2024', '21 November 2024'),
    "check-out date": ('5 April 2024', '15 May 2024', '25 June 2024', '9 July 2024', '18 August 2024', '10 September 2024', '20 October 2024', '28 November 2024'),
    "package carrier": ('FedEx', 'DHL', 'UPS', 'Royal Mail', 'Canada Post', 'Australia Post', 'Japan Post', 'USPS'),
    "tracking number": ('TRK10001', 'TRK20002', 'TRK30003', 'TRK40004', 'TRK50005', 'TRK60006', 'TRK70007', 'TRK80008'),
    "favorite website": ('Wikipedia', 'YouTube', 'GitHub', 'Stack Overflow', 'Reddit', 'LinkedIn', 'Mozilla', 'Khan Academy'),
    "operating system": ('Windows', 'macOS', 'Linux', 'Ubuntu', 'Fedora', 'ChromeOS', 'Android', 'iOS'),
    "device model": ('ThinkPad X1', 'MacBook Air', 'Surface Laptop', 'Dell XPS 13', 'iPad Pro', 'Pixel 8', 'Galaxy S24', 'Framework Laptop'),
    "screen size": ('13 inch', '15 inch', '27 inch', '11 inch', '14 inch', '16 inch', '24 inch', '32 inch'),
    "internet provider": ('Comcast', 'Bell', 'SingTel', 'Verizon', 'AT&T', 'Rogers', 'Vodafone', 'Telstra'),
    "router model": ('Archer C7', 'Eero 6', 'Orbi RBR', 'Nighthawk R7000', 'Google Nest WiFi', 'TP-Link AX55', 'Linksys Velop', 'UniFi Dream Router'),
    "wifi network": ('HomeNet', 'OfficeNet', 'GuestNet', 'LibraryNet', 'StudioNet', 'LabNet', 'CafeNet', 'WorkshopNet'),
    "project name": ('Apollo', 'Beacon', 'Atlas', 'Mercury', 'Horizon', 'Everest', 'Keystone', 'Solstice'),
    "project code": ('PRJ100', 'PRJ200', 'PRJ300', 'PRJ400', 'PRJ500', 'PRJ600', 'PRJ700', 'PRJ800'),
    "team name": ('Red Team', 'Blue Team', 'Green Team', 'Orange Team', 'Yellow Team', 'Purple Team', 'White Team', 'Black Team'),
    "manager name": ('Jordan Smith', 'Taylor Lee', 'Morgan Chen', 'Alex Johnson', 'Jamie Patel', 'Chris Evans', 'Sam Wilson', 'Drew Garcia'),
    "office floor": ('Floor 2', 'Floor 5', 'Floor 8', 'Floor 1', 'Floor 3', 'Floor 4', 'Floor 6', 'Floor 9'),
    "desk number": ('Desk 12', 'Desk 34', 'Desk 56', 'Desk 18', 'Desk 27', 'Desk 45', 'Desk 63', 'Desk 72'),
    "building name": ('Oak Building', 'Maple Building', 'Cedar Building', 'Pine Building', 'River Building', 'Summit Building', 'Sequoia Building', 'Central Building'),
    "meeting room": ('Room Alpha', 'Room Bravo', 'Room Charlie', 'Room Delta', 'Room Echo', 'Room Foxtrot', 'Room Golf', 'Room Hotel'),
    "access level": ('standard', 'privileged', 'administrator', 'read-only', 'editor', 'owner', 'auditor', 'guest'),
    "security question": ('first school', 'first pet', 'birth city', 'favorite teacher', 'childhood address', 'first concert', 'favorite author', 'middle name'),
    "preferred contact method": ('email', 'phone', 'post', 'text message', 'video call', 'messaging app', 'in person', 'mobile app'),
    "customer tier": ('bronze', 'silver tier', 'gold', 'starter', 'standard', 'premium', 'platinum', 'diamond'),
    "subscription plan": ('basic', 'professional', 'enterprise', 'starter', 'team', 'business plus', 'ultimate', 'family'),
    "account status": ('active', 'pending', 'suspended', 'verified', 'locked', 'closed', 'on hold', 'under review'),
}


def position_index(label: str, size: int) -> int:
    """Return the deterministic index for a named target position."""
    if label == "beginning":
        return 0
    if label == "25th_percentile":
        return size // 4
    if label == "50th_percentile":
        return size // 2
    if label == "75th_percentile":
        return (3 * size) // 4
    if label == "end":
        return size - 1
    raise ValueError(f"Unknown position label: {label}")


def value_for(
    key_type: str,
    index: int,
    include_suffix: bool = True,
    base_index: int | None = None,
) -> str:
    """Generate a human-readable value using every eight-value base pool."""
    if key_type == "date of birth":
        if include_suffix:
            return (date(1950, 1, 1) + timedelta(days=index)).strftime("%-d %B %Y")
        selected_index = index if base_index is None else base_index
        return BASE_VALUES[key_type][selected_index % len(BASE_VALUES[key_type])]
    if key_type in KEY_TYPES:
        # Values only need to be unique within one haystack. The generator
        # currently creates at most 300 records per haystack.
        suffix = f"{index % 10000:04d}"
        selected_index = index if base_index is None else base_index
        prefix = BASE_VALUES[key_type][selected_index % len(BASE_VALUES[key_type])]
        return f"{prefix}-{suffix}" if include_suffix else prefix
    raise ValueError(f"Unknown key type: {key_type}")


def shuffled_values(
    keys: list[str],
    example_index: int,
    rng: random.Random,
    include_suffix: bool = True,
    ensure_unique: bool = False,
) -> list[str]:
    """Load shuffled, distinct base values for each key in this haystack."""
    values_by_key: dict[str, list[str]] = {key: [] for key in KEY_TYPES}
    key_counts = {key: keys.count(key) for key in KEY_TYPES}
    base_indices: dict[str, list[int]] = {}
    for key, count in key_counts.items():
        if count == 0:
            continue
        pool_size = len(BASE_VALUES[key])
        indices = []
        while len(indices) < count:
            indices.extend(rng.sample(range(pool_size), pool_size))
        base_indices[key] = indices[:count]

    occurrences = {key: 0 for key in KEY_TYPES}
    for index, key in enumerate(keys):
        values_by_key[key].append(
            value_for(
                key,
                example_index * len(keys) + index,
                include_suffix,
                base_index=base_indices[key][occurrences[key]],
            )
        )
        occurrences[key] += 1
    if ensure_unique:
        present_keys = [key for key in KEY_TYPES if key_counts[key]]
        for _ in range(100):
            used_values: set[str] = set()
            repaired_by_key: dict[str, list[str]] = {}
            for key in present_keys:
                alternatives = list(BASE_VALUES[key])
                rng.shuffle(alternatives)
                available = [
                    value for value in alternatives if value not in used_values
                ]
                count = key_counts[key]
                if len(available) < count:
                    break
                repaired_by_key[key] = available[:count]
                used_values.update(repaired_by_key[key])
            else:
                values_by_key.update(repaired_by_key)
                break
        else:
            raise ValueError("Could not assign globally unique low-similarity values")
    for values in values_by_key.values():
        rng.shuffle(values)
    offsets = {key: 0 for key in KEY_TYPES}
    values = []
    for key in keys:
        values.append(values_by_key[key][offsets[key]])
        offsets[key] += 1
    return values


def make_example(
    size: int,
    condition: str,
    position: str,
    example_index: int,
    seed: int,
) -> dict:
    if condition not in DEFAULT_CONDITIONS:
        raise ValueError(f"Unknown similarity condition: {condition}")
    rng = random.Random(seed)
    subject_numbers = rng.sample(range(10000), size)
    subjects = [f"Subject_{number:04d}" for number in subject_numbers]
    if condition == "high":
        # High similarity rotates through shared attributes only. Identifiers
        # and other one-to-one mappings are excluded from this condition.
        key_type = HIGH_SIMILARITY_KEYS[example_index % len(HIGH_SIMILARITY_KEYS)]
        keys = [key_type] * size
    else:
        # Use every category when possible, with no category appearing more
        # than ceil(size / len(KEY_TYPES)) times.
        keys = [KEY_TYPES[i % len(KEY_TYPES)] for i in range(size)]
    values = shuffled_values(
        keys, example_index, rng, include_suffix=False,
        ensure_unique=condition == "low",
    )
    records = [[subject, key, value]
               for subject, key, value in zip(subjects, keys, values)]
    target_index = rng.randrange(size)
    target = records.pop(target_index)
    target_position = position_index(position, size)
    records.insert(target_position, target)
    target_key = [target[0], target[1]]
    values = [record[2] for record in records]
    if len(subjects) != len(set(subjects)):
        raise AssertionError("Generated subjects are not unique")
    if condition == "low" and len(values) != len(set(values)):
        raise AssertionError("Generated values are not unique")
    if records[target_position] != target:
        raise AssertionError("Target was not placed at the requested position")
    return {
        "example_id": f"{size}_{condition}_{position}_{example_index:04d}",
        "ordered_kv_records": records,
        "target_key": target_key,
        "target_value": target[2],
        "position": target_position,
        "position_label": position,
        "similarity_condition": condition,
        "size": size,
        "seed": seed,
    }


def generate_dataset(
    output_dir: Path,
    sizes: Iterable[int] = DEFAULT_SIZES,
    conditions: Iterable[str] = DEFAULT_CONDITIONS,
    examples_per_position: int = 100,
    seed: int = 42,
    positions: Iterable[str] = POSITIONS,
) -> list[Path]:
    """Write one JSONL file per size/condition/position."""
    if examples_per_position < 1:
        raise ValueError("examples_per_position must be positive")
    positions = tuple(positions)
    if not positions:
        raise ValueError("positions must not be empty")
    condition_offsets = {"high": 17, "low": 31}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for size in sizes:
        for condition in conditions:
            for position in positions:
                path = output_dir / f"kv_dataset_{size}_{condition}_{position}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for index in range(examples_per_position):
                        # Keep all five position variants paired: the seed,
                        # subjects, values, and selected target are identical.
                        # Only the target insertion position varies.
                        example_seed = (seed + size * 1_000_000 + index * 101
                                        + condition_offsets.get(condition, 0))
                        example = make_example(size, condition, position, index, example_seed)
                        handle.write(json.dumps(example, ensure_ascii=True) + "\n")
                paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--examples-per-position", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positions", nargs="+", choices=POSITIONS,
                        default=list(POSITIONS))
    args = parser.parse_args()
    paths = generate_dataset(args.output_dir, examples_per_position=args.examples_per_position,
                             seed=args.seed, positions=args.positions)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
