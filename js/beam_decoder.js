class BeamEntry {
    constructor(parent, label, beam_root) {
        this.parent = parent;
        this.label = label;
        this.children = new Map();
        this.oldp = {
            total: kLogZero,
            blank: kLogZero,
            label: kLogZero
        };
        this.newp = {
            total: kLogZero,
            blank: kLogZero,
            label: kLogZero
        };
    }

    Active() {
        return this.newp.total !== kLogZero;
    }

    GetChild(ind) {
        let child_entry = this.children.get(ind);
        if (!child_entry) {
            child_entry = new BeamEntry(this, ind, beam_root);
            this.children.set(ind, child_entry);
        }
        return child_entry;
    }

    LabelSeq(merge_repeated) {
        let labels = [];
        let prev_label = -1;
        let c = this;
        while (c.parent) {
            if (!merge_repeated || c.label !== prev_label) {
                labels.push(c.label);
            }
            prev_label = c.label;
            c = c.parent;
        }
        return labels.reverse();
    }
}


class CTCBeamSearchDecoder {
    constructor(num_classes, beam_width, scorer, batch_size = 1, merge_repeated = false) {
        this.num_classes = num_classes;
        this.beam_width = beam_width;
        this.scorer = scorer;
        this.batch_size = batch_size;
        this.merge_repeated = merge_repeated;
        this.leaves = new Array(beam_width);
        this.Reset();
    }

    Decode(seq_len, input, output, scores) {
        this.Reset();
        for (let t = 0; t < seq_len; t++) {
            this.Step(input[t]);
        }
        return this.scorer.GetTopPaths(output, scores);
    }

    Step(log_input_t) {
        let new_beam = [];
        for (let hyp of this.beam) {
            for (let idx = 0; idx < this.num_classes; idx++) {
                new_beam.push(hyp.Extend(idx, log_input_t[idx]));
            }
        }
        this.beam = this.scorer.ScoreBeam(new_beam);
    }

    Reset() {
        this.beam = new Array(this.beam_width);
        for (let i = 0; i < this.beam_width; i++) {
            this.beam[i] = new BeamEntry(0, 1.0, []);
        }
    }
}

class DefaultBeamScorer {
    constructor(beam_width) {
        this.beam_width = beam_width;
    }

    ScoreBeam(beam) {
        beam.sort((a, b) => b.log_prob - a.log_prob);
        return beam.slice(0, this.beam_width);
    }

    GetTopPaths(output, scores) {
        // Code to get top paths and scores
    }
}
