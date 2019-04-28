"""
The ratio constraint feature returns not the count of the number of times a constraint is dissatisfied,
but instead the ratio of the number of times it is dissatisfied to the number of times it is satisfied
"""
from string import Template
from itertools import izip
from functools import partial

import torch
import torch.nn.functional as F

from .featurizer import Featurizer
from dataset import AuxTables
from dcparser.constraint import is_symmetric

# unary_template is used for constraints where the current predicate
# used for detecting violations in pos_values have a reference to only
# one relation's (e.g. t1) attribute on one side and a fixed constant
# value on the other side of the comparison.
unary_template = Template('SELECT _vid_, val_id, count(*) violations '
                          'FROM   "$init_table" as t1, $pos_values as t2 '
                          'WHERE  t1._tid_ = t2._tid_ '
                          '  AND  t2.attribute = \'$rv_attr\' '
                          '  AND  $orig_predicates '
                          '  AND  t2.rv_val $operation $rv_val '
                          'GROUP BY _vid_, val_id')

# binary_template is used for constraints where the current predicate
# used for detecting violations in pos_values have a reference to both
# relations (t1, t2) i.e. no constant value in predicate.
binary_template = Template('SELECT _vid_, val_id, count(*) violations '
                           'FROM   "$init_table" as t1, "$init_table" as t2, $pos_values as t3 '
                           'WHERE  t1._tid_ != t2._tid_ '
                           '  AND  $join_rel._tid_ = t3._tid_ '
                           '  AND  t3.attribute = \'$rv_attr\' '
                           '  AND  $orig_predicates '
                           '  AND  t3.rv_val $operation $rv_val '
                           'GROUP BY _vid_, val_id')

# ex_binary_template is used as a fallback for binary_template in case
# binary_template takes too long to query. Instead of counting the # of violations
# this returns simply a 0-1 indicator if the possible value violates the constraint.
ex_binary_template = Template('SELECT _vid_, val_id, 1 violations '
                              'FROM   "$init_table" as $join_rel, $pos_values as t3 '
                              'WHERE  $join_rel._tid_ = t3._tid_ '
                              '  AND  t3.attribute = \'$rv_attr\' '
                              '  AND EXISTS (SELECT $other_rel._tid_ '
                              '              FROM   "$init_table" AS $other_rel '
                              '              WHERE  $join_rel._tid_ != $other_rel._tid_ '
                              '                AND  $orig_predicates '
                              '                AND  t3.rv_val $operation $rv_val)')

opposite_ops = {
    '<>': '=',
    '!=': '=',
    '=' : '<>',
    '<' : '>=',
    '>=': '<',
    '>' : '<=',
    '<=': '>'
}


def gen_feat_tensor(v_counts, total_vars, classes):
    tensor = torch.zeros(total_vars,classes,1)
    if v_counts:
        violations = v_counts[0]
        nonviolations_hash = v_counts[1]
        for entry in violations:
            vid = int(entry[0])
            val_id = int(entry[1]) - 1
            num = float(entry[2])
            if vid not in nonviolations_hash:
                denom = 1
            elif val_id not in nonviolations_hash[vid]:
                denom = 1
            else:
                denom = nonviolations_hash[vid][val_id]

            if denom == 0:
                raise Exception('The fuck?')
            else:
                tensor[vid][val_id][0] = num / (num + denom)
    return tensor


class RatioConstraintFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'RatioConstraintFeaturizer'
        self.constraints = self.ds.constraints
        self.init_table_name = self.ds.raw_data.name

    def create_tensor(self):
        queries, opposite_queries = self.generate_relaxed_sql()
        results = self.ds.engine.execute_queries_w_backup(queries)
        opposite_results = self.ds.engine.execute_queries_w_backup(opposite_queries)
        # We're going to create a hash structure from the opposite results
        o_hash_list = []
        for o in opposite_results:
            o_hash = {}
            for r in o:
                vid, val_id, count = int(r[0]), int(r[1]) - 1, float(r[2])
                if vid in o_hash:
                    o_hash[vid][val_id] = count
                else:
                    o_hash[vid] = {val_id: count}
            o_hash_list.append(o_hash)
            
        zipped = izip(results, o_hash_list)
        tensors = self._apply_func(partial(gen_feat_tensor, total_vars=self.total_vars, classes=self.classes), zipped)
        combined = torch.cat(tensors,2)
        combined = F.normalize(combined, p=2, dim=1)
        return combined

    def generate_relaxed_sql(self):
        query_list = []
        opposite_query_list = []
        for c in self.constraints:
            # Check tuples in constraint
            unary = (len(c.tuple_names) == 1)
            if unary:
                queries, opposite_queries = self.gen_unary_queries(c)
            else:
                queries, opposite_queries = self.gen_binary_queries(c)
            query_list.extend(queries)
            opposite_query_list.extend(opposite_queries)
        return query_list, opposite_query_list

    def execute_queries(self,queries):
        return self.ds.engine.execute_queries_w_backup(queries)

    def relax_unary_predicate(self, predicate):
        """
        relax_unary_predicate returns the attribute, operation, and
        tuple attribute reference.

        :return: (attr, op, const), for example:
            ("StateAvg", "<>", 't1."StateAvg"')
        """
        attr = predicate.components[0][1]
        op = predicate.operation
        if op not in opposite_ops:
            raise Exception('Op does not have matching opposite op')
        opp_op = opposite_ops[op]
        comp = predicate.components[1]
        # do not quote literals/constants in comparison
        const = comp if comp.startswith('\'') else '"{}"'.format(comp)
        return attr, op, opp_op, const

    def relax_binary_predicate(self, predicate, rel_idx):
        """
        relax_binary_predicate returns the attribute, operation, and
        tuple attribute reference.

        :return: (attr, op, const), for example:
            ("StateAvg", "<>", 't1."StateAvg"')
        """
        attr = predicate.components[rel_idx][1]
        op = predicate.operation
        if op not in opposite_ops:
            raise Exception('Op does not have matching opposite op')
        opp_op = opposite_ops[op]
        const = '{}."{}"'.format(
                predicate.components[1-rel_idx][0],
                predicate.components[1-rel_idx][1])

        return attr, op, opp_op, const

    def get_binary_predicate_join_rel(self, predicate):
        if 't1' in predicate.cnf_form and 't2' in predicate.cnf_form:
            if is_symmetric(predicate.operation):
                return True, ['t1'], ['t2']
            else:
                return True, ['t1','t2'], ['t2', 't1']
        elif 't1' in predicate.cnf_form and 't2' not in predicate.cnf_form:
            return False, ['t1'], None
        elif 't1' not in predicate.cnf_form and 't2' in predicate.cnf_form:
            return False, ['t2'], None

    def gen_unary_queries(self, constraint):
        # Iterate over predicates and relax one predicate at a time
        queries = []
        opposite_queries = []
        predicates = constraint.predicates
        for k in range(len(predicates)):
            orig_cnf = self._orig_cnf(predicates, k)
            # If there are no other predicates in the constraint,
            # append TRUE to the WHERE condition. This avoids having
            # multiple SQL templates.
            if len(orig_cnf) == 0:
                orig_cnf = 'TRUE'
            rv_attr, op, opposite_op, rv_val = self.relax_unary_predicate(predicates[k])
            query = unary_template.substitute(init_table=self.init_table_name,
                                              pos_values=AuxTables.pos_values.name,
                                              orig_predicates=orig_cnf,
                                              rv_attr=rv_attr,
                                              operation=op,
                                              rv_val=rv_val)
            opposite_query = unary_template.substitute(init_table=self.init_table_name,
                                              pos_values=AuxTables.pos_values.name,
                                              orig_predicates=orig_cnf,
                                              rv_attr=rv_attr,
                                              operation=opposite_op,
                                              rv_val=rv_val)
            queries.append((query, ''))
            opposite_queries.append((opposite_query, ''))
        return queries, opposite_queries

    def gen_binary_queries(self, constraint):
        queries = []
        opposite_queries = []
        predicates = constraint.predicates
        for k in range(len(predicates)):
            orig_cnf = self._orig_cnf(predicates, k)
            # If there are no other predicates in the constraint,
            # append TRUE to the WHERE condition. This avoids having
            # multiple SQL templates.
            if len(orig_cnf) == 0:
                orig_cnf = 'TRUE'
            is_binary, join_rel, other_rel = self.get_binary_predicate_join_rel(predicates[k])
            if not is_binary:
                rv_attr, op, opposite_op, rv_val = self.relax_unary_predicate(predicates[k])
                query = binary_template.substitute(init_table=self.init_table_name,
                                                   pos_values=AuxTables.pos_values.name,
                                                   join_rel=join_rel[0],
                                                   orig_predicates=orig_cnf,
                                                   rv_attr=rv_attr,
                                                   operation=op,
                                                   rv_val=rv_val)
                opposite_query = binary_template.substitute(init_table=self.init_table_name,
                                                   pos_values=AuxTables.pos_values.name,
                                                   join_rel=join_rel[0],
                                                   orig_predicates=orig_cnf,
                                                   rv_attr=rv_attr,
                                                   operation=opposite_op,
                                                   rv_val=rv_val)
                opposite_queries.append((opposite_query, ''))
                queries.append((query, ''))
            else:
                for idx, rel in enumerate(join_rel):
                    rv_attr, op, opposite_op, rv_val = self.relax_binary_predicate(predicates[k], idx)
                    # count # of queries
                    query = binary_template.substitute(init_table=self.init_table_name,
                                                       pos_values=AuxTables.pos_values.name,
                                                       join_rel=rel,
                                                       orig_predicates=orig_cnf,
                                                       rv_attr=rv_attr,
                                                       operation=op,
                                                       rv_val=rv_val)
                    opposite_query = binary_template.substitute(init_table=self.init_table_name,
                                                       pos_values=AuxTables.pos_values.name,
                                                       join_rel=rel,
                                                       orig_predicates=orig_cnf,
                                                       rv_attr=rv_attr,
                                                       operation=opposite_op,
                                                       rv_val=rv_val)
                    # fallback 0-1 query instead of count
                    ex_query = ex_binary_template.substitute(init_table=self.init_table_name,
                                                             pos_values=AuxTables.pos_values.name,
                                                             join_rel=rel,
                                                             orig_predicates=orig_cnf,
                                                             rv_attr=rv_attr,
                                                             operation=op,
                                                             rv_val=rv_val,
                                                             other_rel=other_rel[idx])
                    opposite_ex_query = ex_binary_template.substitute(init_table=self.init_table_name,
                                                             pos_values=AuxTables.pos_values.name,
                                                             join_rel=rel,
                                                             orig_predicates=orig_cnf,
                                                             rv_attr=rv_attr,
                                                             operation=opposite_op,
                                                             rv_val=rv_val,
                                                             other_rel=other_rel[idx])
                    opposite_queries.append((opposite_query, opposite_ex_query))
                    queries.append((query, ex_query))
        return queries, opposite_queries

    def _orig_cnf(self, predicates, idx):
        """
        _orig_cnf returns the CNF form of the predicates that does not include
        the predicate at index :param idx:.

        This CNF is usually used for the left relation when counting violations.
        """
        orig_preds = predicates[:idx] + predicates[(idx+1):]
        orig_cnf = " AND ".join([pred.cnf_form for pred in orig_preds])
        return orig_cnf

    def feature_names(self):
        return ["fixed pred: {}, violation pred: {}".format(self._orig_cnf(constraint.predicates, idx),
                                                            constraint.predicates[idx].cnf_form)
                for constraint in self.constraints
                for idx in range(len(constraint.predicates))]
