import * as tf from '@tensorflow/tfjs';

import vocab_en from './gpt2/vocab.json';
import nonSpeechTokens_en from './gpt2/non_speech_tokens.json';
import specialTokensMap_en from './gpt2/special_tokens_map.json';

import vocab_ml from './multilingual/vocab.json';
import nonSpeechTokens_ml from './multilingual/non_speech_tokens.json';
import specialTokensMap_ml from './multilingual/special_tokens_map.json';
import addedTokens_ml from './multilingual/added_tokens.json';

const LANGUAGES = {
	en: 'english',
	zh: 'chinese',
	de: 'german',
	es: 'spanish',
	ru: 'russian',
	ko: 'korean',
	fr: 'french',
	ja: 'japanese',
	pt: 'portuguese',
	tr: 'turkish',
	pl: 'polish',
	ca: 'catalan',
	nl: 'dutch',
	ar: 'arabic',
	sv: 'swedish',
	it: 'italian',
	id: 'indonesian',
	hi: 'hindi',
	fi: 'finnish',
	vi: 'vietnamese',
	he: 'hebrew',
	uk: 'ukrainian',
	el: 'greek',
	ms: 'malay',
	cs: 'czech',
	ro: 'romanian',
	da: 'danish',
	hu: 'hungarian',
	ta: 'tamil',
	no: 'norwegian',
	th: 'thai',
	ur: 'urdu',
	hr: 'croatian',
	bg: 'bulgarian',
	lt: 'lithuanian',
	la: 'latin',
	mi: 'maori',
	ml: 'malayalam',
	cy: 'welsh',
	sk: 'slovak',
	te: 'telugu',
	fa: 'persian',
	lv: 'latvian',
	bn: 'bengali',
	sr: 'serbian',
	az: 'azerbaijani',
	sl: 'slovenian',
	kn: 'kannada',
	et: 'estonian',
	mk: 'macedonian',
	br: 'breton',
	eu: 'basque',
	is: 'icelandic',
	hy: 'armenian',
	ne: 'nepali',
	mn: 'mongolian',
	bs: 'bosnian',
	kk: 'kazakh',
	sq: 'albanian',
	sw: 'swahili',
	gl: 'galician',
	mr: 'marathi',
	pa: 'punjabi',
	si: 'sinhala',
	km: 'khmer',
	sn: 'shona',
	yo: 'yoruba',
	so: 'somali',
	af: 'afrikaans',
	oc: 'occitan',
	ka: 'georgian',
	be: 'belarusian',
	tg: 'tajik',
	sd: 'sindhi',
	gu: 'gujarati',
	am: 'amharic',
	yi: 'yiddish',
	lo: 'lao',
	uz: 'uzbek',
	fo: 'faroese',
	ht: 'haitian creole',
	ps: 'pashto',
	tk: 'turkmen',
	nn: 'nynorsk',
	mt: 'maltese',
	sa: 'sanskrit',
	lb: 'luxembourgish',
	my: 'myanmar',
	bo: 'tibetan',
	tl: 'tagalog',
	mg: 'malagasy',
	as: 'assamese',
	tt: 'tatar',
	haw: 'hawaiian',
	ln: 'lingala',
	ha: 'hausa',
	ba: 'bashkir',
	jw: 'javanese',
	su: 'sundanese'
};

const TO_LANGUAGE_CODE = {
	...Object.fromEntries(Object.entries(LANGUAGES).map((a) => a.reverse())),
	// **{language: code for code, language in LANGUAGES.items()},
	burmese: 'my',
	valencian: 'ca',
	flemish: 'nl',
	haitian: 'ht',
	letzeburgesch: 'lb',
	pushto: 'ps',
	panjabi: 'pa',
	moldavian: 'ro',
	moldovan: 'ro',
	sinhalese: 'si',
	castilian: 'es'
};



class Tokenizer {
	constructor(name) {
		if (name == 'multilingual') {
			this.vocab = vocab_ml;
			this.nonSpeechTokens = nonSpeechTokens_ml['non_speech_tokens'];
			this.specialTokensMap = specialTokensMap_ml;
			this.vocab = { ...this.vocab, ...addedTokens_ml };
		} else {
			this.vocab = vocab_en;
			this.nonSpeechTokens = nonSpeechTokens_en['non_speech_tokens'];
			this.specialTokensMap = specialTokensMap_en;
		}
		
		this.name = name;
		
		let allSpecialIds = [];
		for (let key in this.specialTokensMap) {
			allSpecialIds.push(this.vocab[this.specialTokensMap[key]]);
		}

		this.allSpecialIds = [ ...new Set(allSpecialIds) ];
		this.idToToken = Object.fromEntries(Object.entries(this.vocab).map((a) => a.reverse()));
	}

	encodeSingleToken(token) {
		const val = this.vocab[token];
		return val ? val : this.vocab[this.specialTokensMap['unk_token']];
	}
	encode(tokens) {
		let ids = [];
		for (let token of tokens) {
			let val = this.vocab[token];
			if (!val) val = this.vocab[this.specialTokensMap['unk_token']];
			ids.push(val);
		}
		return ids;
	}

	decode(token_ids) {
		let tokens = [];
		for (let id of token_ids) {
			let token = this.idToToken[id];
			if (!token) {
				token = this.specialTokensMap['unk_token'];
			}
			if (token[0] === 'Ġ' && token.length > 1) {
				token = token.slice(1);
			}
			tokens.push(token);
		}
		return tokens;
	}

	addSpecialTokens(specials) {
		this.specialTokensMap['additional_special_tokens'] = specials;
		const all_ids = Object.values(this.vocab);
		let last_id = all_ids[all_ids.length - 1];
		let newSpecialIds = [];
		for (let special_token of specials) {
			last_id += 1;
			this.vocab[special_token] = last_id;
			newSpecialIds.push(last_id);
		}
		this.allSpecialIds = [ ...this.allSpecialIds, ...newSpecialIds ];
		this.idToToken = Object.fromEntries(Object.entries(this.vocab).map((a) => a.reverse()));
		this.eot = this.encodeSingleToken('<|endoftext|>');
		this.sot = this.encodeSingleToken('<|startoftranscript|>');
		this.sotLm = this.encodeSingleToken('<|startoflm|>');
		this.sotPrev = this.encodeSingleToken('<|startofprev|>');
		this.noSpeech = this.encodeSingleToken('<|nospeech|>');
		this.noTimestamps = this.encodeSingleToken('<|notimestamps|>');
	}
}

function buildTokenizer(name) {
	let tokenizer = new Tokenizer(name);
	specials = [
		'<|startoftranscript|>',
		...Object.keys(LANGUAGES).map((lang) => `<|${lang}|>`),
		'<|translate|>',
		'<|transcribe|>',
		'<|startoflm|>',
		'<|startofprev|>',
		'<|nospeech|>',
		'<|notimestamps|>'
	];

	tokenizer.addSpecialTokens(specials);
	return tokenizer;
}

export function getTokenizer(multilingual, task, language) {
	if (language) {
		language = language.toLowerCase();
		if (!(language in LANGUAGES)) {
			if (language in TO_LANGUAGE_CODE) language = TO_LANGUAGE_CODE[language];
			else throw new Error('Incorrect language');
		}
	}
	let tokenizer_name;
	if (multilingual) {
		tokenizer_name = 'multilingual';
		task = task ? task : 'transcribe';
		language = language ? language : 'en';
	} else {
		tokenizer_name = 'gpt2';
		task = null;
		language = null;
	}

	let tokenizer = buildTokenizer(tokenizer_name);

	const allSpecialIds = tokenizer.allSpecialIds;
	const sot = allSpecialIds[1];
	const translate = allSpecialIds[allSpecialIds.length - 6];
	const transcribe = allSpecialIds[allSpecialIds.length - 5];

	const langs = Object.keys(LANGUAGES);
	const sotSequence = [ sot ];

	if (language) sotSequence.push(sot + 1 + langs.indexOf(language));
	if (task) sotSequence.push(task == 'transcribe' ? transcribe : translate);

	tokenizer.sotSequence = sotSequence;
	return tokenizer;
}

