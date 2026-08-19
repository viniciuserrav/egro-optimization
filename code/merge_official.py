"""
merge_official.py — merge per-job partial JSONs from the cec2017_official
workflow into one file per dimension.

Each CI job runs one (dimension, function[, algorithm-group]) slice and
uploads its own cec2017_official_d{dim}.json.  This script deep-merges every
partial found under the given directory (recursively) and writes the three
combined files next to it, printing a completeness report.

Usage:  python merge_official.py [artifacts_dir] [out_dir]
"""
import glob
import json
import os
import sys


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'artifacts'
    dst = sys.argv[2] if len(sys.argv) > 2 else '.'
    merged, conflicts = {}, []

    paths = sorted(glob.glob(os.path.join(src, '**', 'cec2017_official_d*.json'),
                             recursive=True))
    if not paths:
        raise SystemExit('no partial files found under %r' % src)
    for path in paths:
        dim = int(os.path.basename(path).split('_d')[1].split('.')[0])
        with open(path) as fh:
            d = json.load(fh)
        m = merged.setdefault(dim, {'_meta': d.get('_meta', {})})
        for key, rec in d.items():
            if key == '_meta':
                continue
            mrec = m.setdefault(key, {})
            for algo, arec in rec.items():
                if algo not in mrec:
                    mrec[algo] = arec
                    continue
                tgt = mrec[algo]
                for i in range(len(arec['errors'])):
                    if arec['errors'][i] is not None or arec['flags'][i] is not None:
                        prev = tgt['errors'][i]
                        if (prev is not None
                                and arec['errors'][i] is not None
                                and prev != arec['errors'][i]):
                            conflicts.append(
                                '%s/%s run %d: %r vs %r'
                                % (key, algo, i, prev, arec['errors'][i]))
                        tgt['errors'][i] = arec['errors'][i]
                        tgt['evals'][i] = arec['evals'][i]
                        tgt['flags'][i] = arec['flags'][i]
                        if 'xbest' in arec:
                            tgt.setdefault('xbest', [None] * len(arec['errors']))
                            tgt['xbest'][i] = arec['xbest'][i]

    if conflicts:
        print('WARNING: %d conflicting duplicate slots:' % len(conflicts))
        for c in conflicts[:10]:
            print('   ', c)
        raise SystemExit('aborting: partial files disagree on completed runs')

    for dim in sorted(merged):
        d = merged[dim]
        out = os.path.join(dst, 'cec2017_official_d%d.json' % dim)
        with open(out, 'w') as fh:
            json.dump(d, fh)
        total = done = flagged = 0
        for key, rec in d.items():
            if key == '_meta':
                continue
            for algo, arec in rec.items():
                for e, f in zip(arec['errors'], arec['flags']):
                    total += 1
                    if e is not None:
                        done += 1
                    elif f is not None:
                        flagged += 1
        print('%s  slots=%d  done=%d  flagged=%d  missing=%d'
              % (out, total, done, flagged, total - done - flagged))


if __name__ == '__main__':
    main()
